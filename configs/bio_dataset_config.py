visual_datasets = [
    dict(
        type="pmcvqa",
        vis_root="data/pmcvqa/images",
        ann_path="data/pmcvqa/train.csv"
    ),
]

language_datasets = [
    dict(
        type="pubmedqa",
        ann_path="data/pubmedqa/ori_pqal.json"
    )
]