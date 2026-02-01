from python import Python

def load_physical_aiavdataset(
    clip_id: String,
    t0_us: Int = 5100000,
    maybe_stream: Bool = True,
    num_history_steps: Int = 16,
    num_future_steps: Int = 64,
    time_step: Float64 = 0.1,
    num_frames: Int = 4,
) -> PythonObject:
    """Load data from physical_ai_av for model inference (Mojo Implementation)."""
    let np = Python.import_module("numpy")
    let torch = Python.import_module("torch")
    let spt = Python.import_module("scipy.spatial.transform")
    let physical_ai_av = Python.import_module("physical_ai_av")
    let rearrange = Python.import_module("einops").rearrange

    let avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    
    let camera_features = Python.list(
        avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
        avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV
    )

    let camera_name_to_index = Python.dict()
    camera_name_to_index["camera_cross_left_120fov"] = 0
    camera_name_to_index["camera_front_wide_120fov"] = 1
    camera_name_to_index["camera_cross_right_120fov"] = 2
    camera_name_to_index["camera_rear_left_70fov"] = 3
    camera_name_to_index["camera_rear_tele_30fov"] = 4
    camera_name_to_index["camera_rear_right_70fov"] = 5
    camera_name_to_index["camera_front_tele_30fov"] = 6

    # Load egomotion
    let egomotion = avdi.get_clip_feature(clip_id, avdi.features.LABELS.EGOMOTION, maybe_stream=maybe_stream)

    # Compute timestamps
    let time_step_us = int(time_step * 1000000)
    let history_timestamps = np.arange(t0_us - (num_history_steps - 1) * time_step_us, t0_us + 1, time_step_us)
    let future_timestamps = np.arange(t0_us + time_step_us, t0_us + (num_future_steps + 1) * time_step_us, time_step_us)

    # Trajectory processing
    let ego_history = egomotion(history_timestamps)
    let ego_history_xyz = ego_history.pose.translation
    let ego_history_quat = ego_history.pose.rotation.as_quat()

    let ego_future = egomotion(future_timestamps)
    let ego_future_xyz = ego_future.pose.translation
    let ego_future_quat = ego_future.pose.rotation.as_quat()

    let t0_xyz = ego_history_xyz[-1].copy()
    let t0_rot = spt.Rotation.from_quat(ego_history_quat[-1])
    let t0_rot_inv = t0_rot.inv()

    let ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)
    let ego_future_xyz_local = t0_rot_inv.apply(ego_future_xyz - t0_xyz)

    # Convert to Tensors
    let history_xyz_tensor = torch.from_numpy(ego_history_xyz_local).float().unsqueeze(0).unsqueeze(0)
    let future_xyz_tensor = torch.from_numpy(ego_future_xyz_local).float().unsqueeze(0).unsqueeze(0)

    # Camera Loading
    let image_frames_list = Python.list()
    let camera_indices_list = Python.list()
    
    let image_timestamps = np.array([t0_us - (num_frames - 1 - i) * time_step_us for i in range(num_frames)], dtype=np.int64)

    for cam_feat in camera_features:
        let camera = avdi.get_clip_feature(clip_id, cam_feat, maybe_stream=maybe_stream)
        let data = camera.decode_images_from_timestamps(image_timestamps)
        let frames = data[0]
        
        let frames_tensor = rearrange(torch.from_numpy(frames), "t h w c -> t c h w")
        image_frames_list.append(frames_tensor)
        
        # Simple camera name extraction for index
        let cam_name = str(cam_feat).split("/")[-1].lower()
        camera_indices_list.append(camera_name_to_index.get(cam_name, 0))

    let result = Python.dict()
    result["image_frames"] = torch.stack(image_frames_list, dim=0)
    result["camera_indices"] = torch.tensor(camera_indices_list, dtype=torch.int64)
    result["ego_history_xyz"] = history_xyz_tensor
    result["ego_future_xyz"] = future_xyz_tensor
    result["t0_us"] = t0_us
    result["clip_id"] = clip_id

    return result
