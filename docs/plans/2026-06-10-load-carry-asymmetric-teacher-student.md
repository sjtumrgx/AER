# Load-Carry Asymmetric Teacher-Student RL Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Go2 load-carrying asymmetric teacher-student RL path with privileged teacher, proprioceptive-history student encoder, adaptive payload-aware energy regularization, baselines, and metric/plot outputs.

**Architecture:** Extend the existing `ppo_cse` RMA path rather than adding a parallel runner: teacher actor uses current proprioception plus privileged load/terrain state, student actor uses current proprioception plus temporal encoder latent, and the critic remains privileged for training. Add a load-carry Go2 config that exposes the requested privileged payload/friction/terrain/disturbance channels and a reward term for load-normalized CoT weighted by policy-provided adaptive alpha.

**Tech Stack:** Python, PyTorch, Isaac Gym Go2 env, existing `gym_learn/ppo_cse` runner/storage, unittest/pytest smoke tests.

---

### Task 1: Failing architecture tests

**Files:**
- Create: `tests/test_load_carry_teacher_student_architecture.py`

**Steps:**
1. Write tests that instantiate `gym_learn.ppo_cse.ActorCritic` with `num_obs=70`, `num_privileged_obs=40`, `H=30`, latent dim 16.
2. Assert teacher actor first layer consumes `obs + priv`, student actor consumes `obs + latent`, critic consumes `obs + priv`.
3. Assert CNN and GRU encoders accept flattened `H * obs_dim` histories and produce 16-dim latents.
4. Assert adaptive alpha head emits positive scalar weights from latent + command + terrain difficulty.
5. Run the test and verify it fails against the current CSE implementation.

### Task 2: Failing load-carry config/reward/metrics tests

**Files:**
- Create: `tests/test_load_carry_config_reward_metrics.py`

**Steps:**
1. Assert a `LoadCarryGo2Config` exists, uses 20-50 frames of history, defines 40 privileged dimensions, and enables payload mass/CoM/inertia/relative state/friction/terrain/disturbance observation flags.
2. Assert train help exposes `--cfg load_carry`, `--training_stage`, `--baseline`, and `--adaptation_encoder` without importing Isaac Gym during `--help`.
3. Assert load-normalized transport cost divides mechanical energy by `(m_robot + m_payload) g d` and supports none/fixed/adaptive alpha modes.
4. Assert metrics include CoT, tracking error, fall rate, joint power, foot contact schedule, stance duration, body height, and gait transition plotting names.
5. Run tests and verify they fail before implementation.

### Task 3: Implement actor-critic and PPO stage routing

**Files:**
- Modify: `gym_learn/ppo_cse/actor_critic.py`
- Modify: `gym_learn/ppo_cse/ppo.py`
- Modify: `gym_learn/ppo_cse/__init__.py`

**Steps:**
1. Add temporal adaptation encoder classes (CNN/GRU/MLP fallback) with latent dim 8-16.
2. Split teacher actor, student actor, privileged critic, privileged-to-latent target encoder, and alpha head.
3. Add methods for teacher PPO, student distillation, latent supervised pretraining, DAgger action mixing, and student PPO fine-tuning.
4. Keep export names but export student adaptation module plus student actor body for deployment.

### Task 4: Implement load-carry env/config/reward/baselines

**Files:**
- Create: `gym/envs/go2/go2_config_load_carry.py`
- Modify: `gym/envs/base/legged_robot_config.py`
- Modify: `gym/envs/base/legged_robot.py`
- Modify: `gym/envs/rewards/corl_rewards.py`
- Modify: `scripts/train.py`

**Steps:**
1. Add config flags/ranges for payload inertia, dynamic relative pose/velocity, external disturbance, terrain privileged parameters, and baseline modes.
2. Initialize/randomize/load privileged buffers and include them in `compute_observations`.
3. Store policy-provided alpha on the env before `step()` and use it in load-normalized transport reward.
4. Add train CLI routing for teacher, latent pretrain, distill, student PPO, and baselines.

### Task 5: Implement evaluation metrics and plotting hooks

**Files:**
- Modify: `gym_learn/eval_metrics/metrics.py`
- Create or modify: `scripts/evaluate_load_carry.py`
- Modify: `README.md`

**Steps:**
1. Add metric functions for requested payload/dynamic-payload conditions.
2. Add plotting script that emits named PNGs/CSV summaries for CoT, tracking error, fall rate, joint power, contacts, stance, body height, and gait transitions.
3. Document the training/evaluation commands and baseline matrix.

### Task 6: Verification

**Commands:**
- `python -m pytest tests/test_load_carry_teacher_student_architecture.py -q`
- `python -m pytest tests/test_load_carry_config_reward_metrics.py -q`
- `python -m pytest tests/test_go2_neutral_repo.py tests/test_wandb_observability.py -q`
- `python -m compileall -q gym gym_learn scripts tests`
- `python scripts/train.py --help`
- `python scripts/evaluate_load_carry.py --help`
