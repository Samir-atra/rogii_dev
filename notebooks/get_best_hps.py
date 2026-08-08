import json
import os

tuner_dir = 'tuner_results/rogii_dense_opt_norm_augmented_v4'

best_score = float('inf')
best_hps = None
best_trial = None

for fname in os.listdir(tuner_dir):
    if fname.startswith('trial_') and os.path.isdir(os.path.join(tuner_dir, fname)):
        trial_path = os.path.join(tuner_dir, fname, 'trial.json')
        if os.path.exists(trial_path):
            with open(trial_path, 'r') as f:
                trial = json.load(f)
            
            score = trial.get('score')
            if score is not None and score < best_score:
                best_score = score
                best_hps = trial.get('hyperparameters', {}).get('values', {})
                best_trial = fname

print(f"Best trial: {best_trial}")
print(f"Best score: {best_score}")
print(f"Best HPs: {best_hps}")
