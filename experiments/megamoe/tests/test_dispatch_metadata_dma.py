# Copyright (c) 2026, SimpleBright_Man

"""Run the actual metadata DMA helper with independently queued MTE engines."""

from test_comm_split import compile_and_run


def test_dispatch_metadata_dma_pipeline(tmp_path):
    completed = compile_and_run("dispatch_metadata_dma_test.cpp", tmp_path)
    assert "metadata DMA: 1/2/6 slots, peers, route batches, slice drains and full rows passed" in completed.stdout
