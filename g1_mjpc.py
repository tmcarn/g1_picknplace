import mujoco
import mujoco_mpc

# Load your model
model = mujoco.MjModel.from_xml_path("unitree_g1/g1_with_box.xml")
data = mujoco.MjData(model)

# Create MPC planner
planner = mujoco_mpc.agent.Planner(
    model=model,
    task_id='balance',  # or 'walk', 'stand'
    horizon=10,
    planning_timestep=0.01
)

# Run control loop
for i in range(1000):
    # Plan optimal action
    action = planner.plan(data)
    
    # Apply and step
    data.ctrl[:] = action
    mujoco.mj_step(model, data)