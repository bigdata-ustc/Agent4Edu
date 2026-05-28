# RCD-style KCG Builder for Agent4Edu

Agent4Edu does **not** use the full RCD cognitive diagnosis model.  It only uses the concept-map construction utility released in RCD under `data/ASSIST/graph` to build a knowledge concept graph (KCG) for memory reinforcement.

This folder refactors that utility into an Agent4Edu tool.  Compared with the original scripts, it:

- supports Agent4Edu `stu_logs.json` directly;
- supports the original RCD `log_data_all.json` structure;
- preserves raw Agent4Edu knowledge ids by default;
- writes `knowledgeGraph.txt`, `K_Directed.txt`, `K_Undirected.txt`, `kcg.json`, and `graph_stats.json` in one command;
- does not depend on DGL or the RCD model.

## Build KCG from Agent4Edu logs

```bash
python Code/tools/rcd_graph/build_kcg.py \
  --logs data/iflytek_sample/stu_logs.json \
  --output-dir data/iflytek_sample/graph \
  --agent-kcg data/iflytek_sample/kcg.json \
  --relation-scope all
```

## Reproduce the original RCD ASSIST indexing style

The original RCD ASSIST script subtracts 1 from concept ids.  Use:

```bash
python Code/tools/rcd_graph/build_kcg.py \
  --logs path/to/log_data_all.json \
  --output-dir path/to/graph \
  --knowledge-id-offset -1
```

## Relation scope

- `all`: exports all selected concept-map edges to Agent4Edu `kcg.json`; this matches the current Agent4Edu memory behavior, where any selected relation means two concepts are related.
- `undirected`: exports reciprocal/similarity relations only.
- `directed`: exports one-way dependency relations only.

## Citation

The graph construction logic is adapted from:

```bibtex
@inproceedings{gao2021rcd,
  title={RCD: Relation map driven cognitive diagnosis for intelligent education systems},
  author={Gao, Weibo and Liu, Qi and Huang, Zhenya and Yin, Yu and Bi, Haoyang and Wang, Mu-Chun and Ma, Jianhui and Wang, Shijin and Su, Yu},
  booktitle={Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval},
  pages={501--510},
  year={2021}
}
```

Repository: https://github.com/bigdata-ustc/RCD
