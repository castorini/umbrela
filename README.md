# umBRELA

## 📟 Instructions

### Create Conda Environment

```bash
conda create -n umbrela python=3.10
conda activate umbrela
```

### If you do not already have JDK 21 installed, install via conda:
```bash
conda install -c conda-forge openjdk=21 maven -y
```

### Install following dependencies for retrieval:
```bash
conda install -c pytorch faiss-cpu pytorch -y
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Judgment generation snippet

#### Setting up the model jugde:
```python
from umbrela.vicuna_judge import VicunaJudge

judge_vicuna = VicunaJudge("dl19-passage")
```

#### Passing qrel-passages for evaluations:
```python
input_dict = {
    "query": {"text": "how long is life cycle of flea", "qid": "264014"},
    "candidates": [
        {
            "doc": {
                "segment": "The life cycle of a flea can last anywhere from 20 days to an entire year. It depends on how long the flea remains in the dormant stage (eggs, larvae, pupa). Outside influences, such as weather, affect the flea cycle. A female flea can lay around 20 to 25 eggs in one day."
            },
            "docid": "4834547",
            "score": 14.971799850463867,
        },
    ]
}

judgments = judge_vicuna.judge(input_dict)
```


## 🙏 Acknowledgments

This research is supported in part by the Natural Sciences and Engineering Research Council (NSERC) of Canada.