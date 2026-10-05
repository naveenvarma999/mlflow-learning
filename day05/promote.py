from pathlib import Path
import mlflow
from mlflow import MlflowClient

DB = Path(__file__).parent / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{DB}")

client = MlflowClient()
MODEL = "iris-classifier"
METRIC = "accuracy"

def accuracy_of(version):
    run = client.get_run(version.run_id)
    return run.data.metrics[METRIC]

# the newest version is the candidate
versions = client.search_model_versions(f"name='{MODEL}'")
candidate = max(versions, key=lambda v: int(v.version))
candidate_acc = accuracy_of(candidate)

print(f"candidate: version {candidate.version}, {METRIC}={candidate_acc}")


# is there a current champion?
try:
    champion = client.get_model_version_by_alias(MODEL, "champion")
    champion_acc = accuracy_of(champion)
    print(f"champion : version {champion.version}, {METRIC}={champion_acc}")
except mlflow.exceptions.MlflowException:
    champion = None
    champion_acc = None
    print("champion : none yet")



# the decision
if champion is None:
    promote, reason = True, "no existing champion"
elif candidate.version == champion.version:
    promote, reason = False, "candidate is already champion"
elif candidate_acc > champion_acc:
    promote, reason = True, f"{candidate_acc:.4f} > {champion_acc:.4f}"
else:
    promote, reason = False, f"{candidate_acc:.4f} not better than {champion_acc:.4f}"



# act, and record why
if promote:
    client.set_registered_model_alias(MODEL, "champion", candidate.version)
    client.set_model_version_tag(MODEL, candidate.version, "promotion", "accepted")
    print(f"PROMOTED version {candidate.version} — {reason}")
else:
    client.set_model_version_tag(MODEL, candidate.version, "promotion", "rejected")
    print(f"REJECTED version {candidate.version} — {reason}")

client.set_model_version_tag(MODEL, candidate.version, "promotion_reason", reason)



