import os
import tempfile
import zipfile
from pathlib import Path

from comet_ml.offline import OfflineExperiment


os.environ["COMET_DISABLE_ANNOUNCEMENT"] = "1"
os.environ["COMET_DISABLE_AUTO_LOGGING"] = "1"

offline_dir = tempfile.mkdtemp(prefix="comet-offline-")
experiment = OfflineExperiment(
    project_name="conda-forge-test",
    offline_directory=offline_dir,
    auto_output_logging=False,
    auto_param_logging=False,
    auto_metric_logging=False,
    log_code=False,
    log_graph=False,
    parse_args=False,
    log_env_details=False,
    log_env_gpu=False,
    log_env_cpu=False,
    log_env_network=False,
    log_env_disk=False,
    log_git_metadata=False,
    log_git_patch=False,
    display_summary_level=0,
)
experiment.set_name("conda-forge-smoke")
experiment.log_parameter("learning_rate", 0.01)
experiment.log_metric("accuracy", 0.75, step=1)
experiment.end()

archives = list(Path(offline_dir).glob("*.zip"))
assert len(archives) == 1, archives

with zipfile.ZipFile(archives[0]) as archive:
    archive_names = set(archive.namelist())
    assert {"experiment.json", "messages.json"} <= archive_names
    messages = archive.read("messages.json").decode()

assert "learning_rate" in messages
assert "accuracy" in messages
