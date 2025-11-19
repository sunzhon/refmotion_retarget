from utils.gmr import GeneralMotionRetargeting as GMR

def test():
    actual_human_height =1.5
    tgt_robot = "lumos_lus2"

    # Initialize the retargeting system
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=tgt_robot,
        use_velocity_limit=False
    )
    
    qpos_list = []
    human_motion_data = []
    # Start the viewer
    smplx_data = {"pelvis": , ""}
    # retarget
    qpos = retarget.retarget(smplx_data, False)# offset ground 

    return qpos
