# MPC-Guided RL for G1 Humanoid

## Overview

This implementation combines **Model Predictive Control (MPC)** with **Reinforcement Learning (RL)** to create robust control policies for the G1 humanoid robot. The approach addresses a common problem in pure RL: difficulty discovering complex behaviors from scratch.

## Architecture

### 1. MPC Controller (`SimpleMPC` class)
- **Purpose**: Generates reference actions based on the robot model
- **Implementation**: PD controller toward standing pose (can be extended to full optimization)
- **Advantages**:
  - Model-based, guarantees feasible actions
  - Fast computation
  - Provides structured exploration signal

### 2. RL Policy (PPO)
- **Purpose**: Learns corrections to MPC actions
- **Action Space**: Outputs **corrections** to MPC reference actions
- **Advantages**:
  - Compensates for model mismatch
  - Adapts to unmodeled dynamics
  - Can improve upon MPC through learning

### 3. Combined Control
```
final_action = mpc_reference_action + rl_correction
```

## Observation Space

The observation includes both state and MPC information:
```
[qpos, qvel, mpc_reference_action, tracking_error]
```

- **qpos**: Joint positions (29 DOF)
- **qvel**: Joint velocities (28 DOF)
- **mpc_reference_action**: Current MPC output (43 actuators)
- **tracking_error**: Previous tracking error (43 actuators)

**Total**: ~143 dimensions

## Reward Function

The reward balances tracking and task objectives:

1. **Tracking Reward** (-1.0 × tracking_error)
   - Encourages following MPC reference
   - Prevents RL from ignoring MPC completely

2. **Task Rewards**
   - Height reward (5.0 weight): Target 0.79m
   - Upright reward (5.0 weight): Maintain vertical orientation

3. **Efficiency Penalty** (-0.01 × correction²)
   - Penalizes excessive RL corrections
   - Encourages relying on MPC when possible

4. **Stability** (-0.01 × velocity²)
   - Smooth motion

5. **Alive Bonus** (+1.0)
   - Survival incentive

## Training Strategy

### Phase 1: Learn to Track (Current)
- MPC provides standing reference
- RL learns to track with minimal corrections
- Builds trust in MPC guidance

### Phase 2: Task Adaptation (Future)
- Introduce task-specific objectives
- RL learns when to deviate from MPC
- Example: Walking gait, manipulation tasks

### Phase 3: Online Adaptation (Future)
- RL adapts MPC parameters online
- Handles novel situations
- Sim-to-real transfer

## Key Differences from Pure RL

| Aspect | Pure RL | MPC-Guided RL |
|--------|---------|---------------|
| Exploration | Random/entropy-driven | Structured by MPC |
| Sample Efficiency | Low (needs many samples) | Higher (MPC guides) |
| Initial Behavior | Poor/random | Reasonable (from MPC) |
| Final Performance | Can be optimal | MPC baseline + RL improvement |
| Sim-to-Real | Difficult | Easier (MPC provides structure) |

## Files

- **`g1_mpc_guided_env.py`**: Environment with MPC + RL control
- **`train_mpc_guided.py`**: Training script
- **`test_mpc_guided.py`**: Testing/visualization script

## Usage

### Training
```bash
python train_mpc_guided.py
```

Training uses 4 parallel environments and runs for 200k timesteps (~50k iterations).

### Testing
```bash
python test_mpc_guided.py --model rl_models/g1_mpc_guided_XX_XX:XX:XX/best_model.zip
```

### Monitoring
```bash
tensorboard --logdir rl_logs/
```

## Expected Behavior

### During Training
- **Early**: RL makes large corrections, high tracking error
- **Middle**: RL learns to trust MPC, corrections decrease
- **Late**: RL makes small, precise corrections for improved performance

### Metrics to Watch
- **Tracking Error**: Should decrease over time
- **Task Reward**: Should increase (height + upright)
- **Correction Magnitude**: Should stabilize at small values

## Extending to Walking

To extend this approach to walking:

1. **Update MPC Reference**
   ```python
   def compute_walking_reference(self, data, target_velocity):
       # Compute foot placement targets
       # Generate swing/stance trajectory
       # Return periodic reference actions
   ```

2. **Add Gait Tracking Rewards**
   ```python
   # Reward for matching desired velocity
   velocity_reward = -abs(actual_vel - target_vel)
   
   # Reward for proper foot clearance
   clearance_reward = swing_foot_height if in_swing_phase else 0
   ```

3. **Phase-Based References**
   - Use gait phase to switch between swing and stance references
   - MPC generates different actions per phase

## Theory: Why This Works

### 1. **Warm Start**
MPC provides a reasonable starting policy, avoiding the "cold start" problem of pure RL.

### 2. **Structured Exploration**
Instead of random actions, exploration happens in the space of "corrections to sensible actions."

### 3. **Safety**
MPC bounds ensure actions stay feasible even during early training.

### 4. **Scalability**
As tasks become more complex, MPC provides increasingly valuable guidance.

### 5. **Interpretability**
Can analyze what RL learned by examining when/how it corrects MPC.

## Limitations & Future Work

### Current Limitations
1. **Simple MPC**: Using PD control instead of full optimization
2. **Standing Only**: Not yet extended to dynamic tasks
3. **No Online Adaptation**: MPC parameters are fixed

### Future Extensions
1. **Full MPC Optimization**
   - Implement nonlinear MPC with contact constraints
   - Use differentiable physics for fast gradient computation

2. **Hierarchical Control**
   - High-level RL for task planning
   - MPC for local trajectory optimization
   - Low-level RL for tracking/adaptation

3. **Domain Randomization**
   - Train RL to handle model uncertainty
   - Robust to parameter variations

4. **Sim-to-Real Transfer**
   - Use MPC-guidance to bridge reality gap
   - RL learns sim-to-real residuals

## References

This implementation is inspired by:
- "Multi-contact MPC for Dynamic Loco-manipulation on Humanoid Robots"
- Residual RL approaches (e.g., residual policy learning)
- MPC + RL hybrid methods in legged robotics

## Comparison to Paper Approach

The shared paper likely describes a more sophisticated approach with:
- Multi-contact constraints in MPC
- Whole-body dynamics optimization
- Contact force planning for manipulation

Our implementation is a simplified version focusing on:
- Core MPC + RL concept
- Standing as a foundational task
- Extensible architecture for future work

To fully implement the paper's approach, we would need:
1. Multi-contact MPC solver
2. Manipulation task environments
3. Contact-aware reward shaping
4. Hierarchical control layers
