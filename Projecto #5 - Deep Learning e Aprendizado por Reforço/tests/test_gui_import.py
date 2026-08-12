def test_gui_module_imports():
    from ntxuva.gui.app import NtxuvaApp, run_gui
    assert NtxuvaApp is not None
    assert callable(run_gui)
