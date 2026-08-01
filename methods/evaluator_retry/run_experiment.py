#!/usr/bin/env python3
"""Evaluator-guided retry baseline for robust rare-prompt image generation.

Each attempt is scored for prompt alignment, explicit requirements, and basic
image usability. Images below the configured threshold are regenerated using
an emphasis prompt derived from the failed requirements.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("experiments/current/evaluator_retry"), type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def normalized_clip_similarity(model, processor, image, texts, device):
    """Return cosine similarity in [0, 1] for one image and one or more texts."""
    inputs = processor(text=texts, images=[image], return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
        image_features = torch.nn.functional.normalize(outputs.image_embeds.float(), dim=-1)
        text_features = torch.nn.functional.normalize(outputs.text_embeds.float(), dim=-1)
    similarities = image_features @ text_features.T
    return ((similarities[0].detach().cpu().numpy() + 1.0) / 2.0).astype(float)


def image_usability(image: Image.Image) -> float:
    """A lightweight proxy for unusably flat, dark, or blurry outputs."""
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    luminance = pixels @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    exposure = float(np.clip(1.0 - abs(luminance.mean() - 0.5) / 0.5, 0.0, 1.0))
    contrast = float(np.clip(luminance.std() / 0.25, 0.0, 1.0))
    edge_strength = np.mean(np.abs(np.diff(luminance, axis=0))) + np.mean(np.abs(np.diff(luminance, axis=1)))
    sharpness = float(np.clip(edge_strength / 0.20, 0.0, 1.0))
    return 0.35 * exposure + 0.30 * contrast + 0.35 * sharpness


def evaluate(model, processor, image, config, device):
    requirements = config["requirements"]
    similarities = normalized_clip_similarity(
        model, processor, image, [config["rare_prompt"], *[item["text"] for item in requirements]], device
    )
    alignment = float(similarities[0])
    requirement_scores = similarities[1:]
    requirement_score = float(np.average(requirement_scores, weights=[item.get("weight", 1.0) for item in requirements]))
    usability = image_usability(image)
    weights = config["scoring_weights"]
    total = weights["alignment"] * alignment + weights["requirements"] * requirement_score + weights["usability"] * usability
    failed = [item["name"] for item, score in zip(requirements, requirement_scores)
              if score < item.get("minimum_score", config["requirement_minimum_score"])]
    return {"total_score": float(total), "alignment_score": alignment, "requirements_score": requirement_score,
            "usability_score": usability, "requirement_scores": {item["name"]: float(score) for item, score in zip(requirements, requirement_scores)},
            "failed_requirements": failed}


def revised_prompt(config, failed_requirements):
    if not failed_requirements:
        return config["rare_prompt"]
    requirements = {item["name"]: item for item in config["requirements"]}
    emphases = [requirements[name].get("emphasis", requirements[name]["text"]) for name in failed_requirements]
    return f"{config['rare_prompt']}. Clearly show: {'; '.join(emphases)}. {config['retry_suffix']}"


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.device == "cpu":
        raise RuntimeError("A CUDA GPU is required for this Stable Diffusion experiment.")
    root = args.output_dir / config["experiment_name"]
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing experiment output: {root}. "
            "Use a new experiment_name or --output-dir."
        )
    attempts_dir, selected_dir = root / "attempts", root / "selected"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    selected_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(config["model_id"], torch_dtype=torch.float16, safety_checker=None).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    evaluator = CLIPModel.from_pretrained(config["evaluator_model_id"]).to(args.device).eval()
    processor = CLIPProcessor.from_pretrained(config["evaluator_model_id"])
    generation_args = {"negative_prompt": config["negative_prompt"], "num_inference_steps": config["num_inference_steps"],
                       "guidance_scale": config["guidance_scale"], "height": config["height"], "width": config["width"]}

    rows, selections = [], []
    for base_seed in tqdm(config["seeds"][:config["num_images"]], desc="Evaluator-guided generation"):
        prompt, best = config["rare_prompt"], None
        attempts_used = 0
        for attempt in range(config["max_attempts"]):
            attempts_used = attempt + 1
            seed = base_seed + attempt * config["retry_seed_stride"]
            image = pipe(prompt=prompt, generator=torch.Generator(device=args.device).manual_seed(seed), **generation_args).images[0]
            report = evaluate(evaluator, processor, image, config, args.device)
            image_path = attempts_dir / f"base_{base_seed}_attempt_{attempt + 1}.png"
            image.save(image_path)
            row = {"base_seed": base_seed, "attempt": attempt + 1, "seed": seed, "prompt": prompt, "image": str(image_path),
                   **{key: value for key, value in report.items() if key != "requirement_scores"}}
            row.update({f"requirement_{name}": score for name, score in report["requirement_scores"].items()})
            rows.append(row)
            if best is None or report["total_score"] > best["report"]["total_score"]:
                best = {"image": image, "report": report, "attempt": attempt + 1, "seed": seed, "prompt": prompt}
            if report["total_score"] >= config["acceptance_threshold"] and not report["failed_requirements"]:
                break
            prompt = revised_prompt(config, report["failed_requirements"])

        selected_path = selected_dir / f"seed_{base_seed}.png"
        best["image"].save(selected_path)
        accepted = best["report"]["total_score"] >= config["acceptance_threshold"] and not best["report"]["failed_requirements"]
        selections.append({"base_seed": base_seed, "selected_attempt": best["attempt"], "selected_seed": best["seed"],
                           "attempts_used": attempts_used, "accepted": accepted, "selected_image": str(selected_path),
                           "selected_prompt": best["prompt"], **best["report"]})

    pd.DataFrame(rows).to_csv(root / "attempt_metrics.csv", index=False)
    pd.DataFrame(selections).to_csv(root / "selected_metrics.csv", index=False)
    selected_scores = [item["total_score"] for item in selections]
    baseline_rows = [item for item in rows if item["attempt"] == 1]
    baseline_scores = [item["total_score"] for item in baseline_rows]
    accepted_outputs = int(sum(item["accepted"] for item in selections))
    summary = {"config": config, "num_outputs": len(selections), "mean_selected_score": float(np.mean(selected_scores)),
               "mean_baseline_score": float(np.mean(baseline_scores)),
               "mean_score_delta_vs_baseline": float(np.mean(np.asarray(selected_scores) - np.asarray(baseline_scores))),
               "accepted_outputs": accepted_outputs, "acceptance_rate": accepted_outputs / len(selections),
               "mean_attempts_used": float(np.mean([item["attempts_used"] for item in selections])),
               "mean_retries": float(np.mean([item["attempts_used"] - 1 for item in selections]))}
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
