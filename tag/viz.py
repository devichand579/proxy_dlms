"""Write a self-contained HTML viewer (vis-network) for a TAG ``graph.json``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def graph_dict_to_vis_format(g: Dict[str, Any]) -> Dict[str, Any]:
    """Convert ``graph_to_dict`` payload to vis.js ``nodes`` / ``edges`` lists."""
    nodes_out = []
    for n in g.get("nodes", []):
        title = n.get("label", n["id"])
        color = {
            "feat": "#a6d8ff",
            "in": "#b5f5c8",
            "out": "#ffb3b3",
            "err": "#ffd6a0",
        }.get(n.get("kind", "feat"), "#ddd")
        nodes_out.append(
            {
                "id": n["id"],
                "label": str(n.get("id", ""))[:48],
                "title": title,
                "group": n.get("kind", "feat"),
                "color": color,
            }
        )
    edges_out = []
    for e in g.get("edges", []):
        w = float(e.get("weight", 0.0))
        wid = min(5.0, max(0.3, 1.0 + 0.1 * (abs(w) ** 0.5)))
        edges_out.append(
            {
                "from": e["source"],
                "to": e["target"],
                "value": abs(w) + 1e-6,
                "title": f"w={w:.4g}",
                "width": wid,
            }
        )
    return {"nodes": nodes_out, "edges": edges_out}


def write_tag_html(
    graph: Dict[str, Any],
    out_path: str | Path,
    *,
    title: str = "TAG viewer",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis = graph_dict_to_vis_format(graph)
    data_json = json.dumps(vis, indent=0)
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style> body {{ margin:0; font-family: system-ui, sans-serif; }} #g {{ width: 100vw; height: 100vh; border:1px solid #ccc; }} </style>
</head>
<body>
  <div id="g"></div>
  <script>
  const data = {data_json};
  const container = document.getElementById("g");
  const nset = new vis.DataSet(data.nodes);
  const eset = new vis.DataSet(data.edges);
  const options = {{ physics: {{ enabled: true, barnesHut: {{ gravitationalConstant: -8000, springLength: 80 }} }} }};
  new vis.Network(container, {{ nodes: nset, edges: eset }}, options);
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def write_graph_json(
    graph: Dict[str, Any],
    out_path: str | Path,
) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
