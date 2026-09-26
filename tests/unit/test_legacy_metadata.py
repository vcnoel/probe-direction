from __future__ import annotations

import pandas as pd
import pytest

from eval_awareness_spectral.analysis import cells_in, require_cell_metadata


def test_untagged_legacy_rows_fail_instead_of_being_relabelled() -> None:
    untagged = pd.DataFrame({"metric": [0.1]})
    with pytest.raises(ValueError, match="random-walk basis metrics"):
        require_cell_metadata(untagged)
    with pytest.raises(ValueError, match="must be recomputed"):
        cells_in(untagged)

    tagged = pd.DataFrame(
        {
            "graph_scope": ["task_subgraph"],
            "normalization": ["sym"],
        }
    )
    require_cell_metadata(tagged)
    assert cells_in(tagged) == [("task_subgraph", "sym")]
    assert cells_in(pd.DataFrame()) == []
