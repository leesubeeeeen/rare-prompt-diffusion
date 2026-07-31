# Linear distribution interpolation (closed baseline)

구성요소별 CLIP text-conditioning에 shrinkage Gaussian을 적합하고, 분포 peak의 선형 중심 주변 후보 중 결합 log-density가 높은 conditioning을 선택합니다.

이 방법은 token embedding을 위치별로 혼합하면서 명사-수식어 관계와 문장 구조를 잃기 때문에 주 연구 방법으로는 종료했습니다. 코드는 재현 및 향후 ablation을 위해 보존합니다.
