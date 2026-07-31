#!/usr/bin/env python3
"""Linear valid-distribution baseline for rare-prompt diffusion generation."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from sklearn.covariance import LedoitWolf
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


TEMPLATES = [
    "a detailed photograph of {term}",
    "a cinematic illustration of {term}",
    "a beautiful scene featuring {term}",
    "highly detailed concept art of {term}",
    "an atmospheric image of {term}",
    "a realistic depiction of {term}",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("outputs"), type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def make_component_prompts(component, count):
    prompts = []
    rng = random.Random(2026 + sum(map(ord, component["name"])))
    for index in range(count):
        term = component["terms"][index % len(component["terms"])]
        prompts.append(rng.choice(TEMPLATES).format(term=term))
    return prompts


@torch.inference_mode()
def encode_prompts(pipe, prompts, device):
    """Return SD's full [77, hidden] conditioning tensors and pooled normalized vectors."""
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    batch = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length,
                      truncation=True, return_tensors="pt")
    ids = batch.input_ids.to(device)
    hidden = text_encoder(ids)[0].float()
    pooled = hidden.mean(dim=1)
    pooled = torch.nn.functional.normalize(pooled, dim=1)
    return hidden, pooled


def gaussian_log_density(samples, mean, precision):
    delta = samples - mean[None, :]
    return -0.5 * np.einsum("bi,ij,bj->b", delta, precision, delta)


def fit_distributions(component_pooled):
    distributions = []
    for pooled in component_pooled:
        values = pooled.cpu().numpy()
        estimator = LedoitWolf().fit(values)
        distributions.append({
            "mean": estimator.location_.astype(np.float32),
            "precision": estimator.precision_.astype(np.float32),
        })
    return distributions


def choose_valid_linear_candidate(component_hidden, component_pooled, distributions, num_candidates, center_weight):
    """Sample convex combinations near equal weights and select the high-density candidate."""
    n_components = len(component_hidden)
    rng = np.random.default_rng(2026)
    weights = rng.dirichlet(np.full(n_components, 18.0), size=num_candidates).astype(np.float32)
    weights[0] = 1.0 / n_components  # explicitly include the exact linear center

    peaks = np.stack([item["mean"] for item in distributions])
    center = peaks.mean(axis=0)
    candidates_pooled = []
    candidates_hidden = []
    for weight in weights:
        # Candidate 0 is precisely the linear center connecting component peaks.
        # The remaining candidates represent nearby, observed-distribution variants.
        if len(candidates_hidden) == 0:
            hidden = sum(float(weight[i]) * component_hidden[i].mean(dim=0) for i in range(n_components))
        else:
            sampled_indices = [rng.integers(len(hidden)) for hidden in component_hidden]
            hidden = sum(float(weight[i]) * component_hidden[i][sampled_indices[i]] for i in range(n_components))
        pooled = torch.nn.functional.normalize(hidden.mean(dim=0), dim=0)
        candidates_hidden.append(hidden)
        candidates_pooled.append(pooled.cpu().numpy())

    candidate_array = np.stack(candidates_pooled)
    density = sum(gaussian_log_density(candidate_array, d["mean"], d["precision"]) for d in distributions)
    distance = ((candidate_array - center[None, :]) ** 2).sum(axis=1)
    scores = density - center_weight * distance
    best = int(np.argmax(scores))
    records = [{
        "index": int(i), "weights": weights[i].round(5).tolist(),
        "density_score": float(density[i]), "center_distance": float(distance[i]), "total_score": float(scores[i]),
    } for i in range(num_candidates)]
    return candidates_hidden[best].unsqueeze(0), records, best


@torch.inference_mode()
def clip_score(model, processor, image, text, device):
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    return float(outputs.logits_per_image[0, 0].detach().cpu())


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.device == "cpu":
        raise RuntimeError("Diffusion image generation requires a CUDA GPU for this experiment.")
    dtype = torch.float16
    pipe = StableDiffusionPipeline.from_pretrained(config["model_id"], torch_dtype=dtype, safety_checker=None)
    pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)

    root = args.output_dir / config["experiment_name"]
    (root / "baseline").mkdir(parents=True, exist_ok=True)
    (root / "linear_valid").mkdir(parents=True, exist_ok=True)
    component_hidden, component_pooled, component_prompts = [], [], {}
    for component in config["components"]:
        prompts = make_component_prompts(component, config["num_prompts_per_term"])
        hidden, pooled = encode_prompts(pipe, prompts, args.device)
        component_hidden.append(hidden)
        component_pooled.append(pooled)
        component_prompts[component["name"]] = prompts

    distributions = fit_distributions(component_pooled)
    selected_embed, candidate_records, selected_index = choose_valid_linear_candidate(
        component_hidden, component_pooled, distributions, config["num_candidates"], config["center_weight"]
    )
    selected_embed = selected_embed.to(device=args.device, dtype=dtype)
    negative_embed, _ = encode_prompts(pipe, [config["negative_prompt"]], args.device)
    negative_embed = negative_embed.to(dtype=dtype)
    (root / "component_prompts.json").write_text(json.dumps(component_prompts, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "candidates.json").write_text(json.dumps(candidate_records, indent=2), encoding="utf-8")

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(args.device).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    rows = []
    for seed in tqdm(config["seeds"][:config["num_images_per_method"]], desc="Generating"):
        common = dict(num_inference_steps=config["num_inference_steps"], guidance_scale=config["guidance_scale"],
                      height=config["height"], width=config["width"])
        generator = torch.Generator(device=args.device).manual_seed(seed)
        baseline = pipe(prompt=config["rare_prompt"], negative_prompt=config["negative_prompt"], generator=generator, **common).images[0]
        baseline_path = root / "baseline" / f"seed_{seed}.png"
        baseline.save(baseline_path)
        rows.append({"method": "baseline", "seed": seed, "image": str(baseline_path),
                     "clip_score": clip_score(clip_model, clip_processor, baseline, config["rare_prompt"], args.device)})

        generator = torch.Generator(device=args.device).manual_seed(seed)
        linear_image = pipe(prompt_embeds=selected_embed, negative_prompt_embeds=negative_embed,
                            generator=generator, **common).images[0]
        linear_path = root / "linear_valid" / f"seed_{seed}.png"
        linear_image.save(linear_path)
        rows.append({"method": "linear_valid", "seed": seed, "image": str(linear_path),
                     "clip_score": clip_score(clip_model, clip_processor, linear_image, config["rare_prompt"], args.device)})

    metrics = pd.DataFrame(rows)
    metrics.to_csv(root / "metrics.csv", index=False)
    summary = {"config": config, "selected_candidate_index": selected_index,
               "selected_candidate": candidate_records[selected_index],
               "mean_clip_score": metrics.groupby("method")["clip_score"].mean().to_dict()}
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
