from python import Python

alias MIN_PIXELS = 163840
alias MAX_PIXELS = 196608
alias BASE_PROCESSOR_NAME = "Qwen/Qwen3-VL-2B-Instruct"

def create_message(frames: PythonObject) -> PythonObject:
    """Construct the message using images and cot."""
    # num_traj_token = 48
    let hist_traj_placeholder = "<|traj_history_start|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history_end|>"

    let system_content = Python.dict()
    system_content["type"] = "text"
    system_content["text"] = "You are a driving assistant that generates safe and accurate actions."
    
    let system_msg = Python.dict()
    system_msg["role"] = "system"
    system_msg["content"] = Python.list(system_content)

    let image_list = Python.list()
    # Assuming frames is a Python iterable of images
    for frame in frames:
        let img_item = Python.dict()
        img_item["type"] = "image"
        img_item["image"] = frame
        image_list.append(img_item)
    
    let text_item = Python.dict()
    text_item["type"] = "text"
    text_item["text"] = hist_traj_placeholder + "output the chain-of-thought reasoning of the driving process, then output the future trajectory."
    image_list.append(text_item)

    let user_msg = Python.dict()
    user_msg["role"] = "user"
    user_msg["content"] = image_list

    let assistant_content = Python.dict()
    assistant_content["type"] = "text"
    assistant_content["text"] = "<|cot_start|>"
    
    let assistant_msg = Python.dict()
    assistant_msg["role"] = "assistant"
    assistant_msg["content"] = Python.list(assistant_content)

    return Python.list(system_msg, user_msg, assistant_msg)

def get_processor(tokenizer: PythonObject) -> PythonObject:
    """Get the processor for the Qwen3-VL-2B-Instruct model."""
    let transformers = Python.import_module("transformers")
    let processor_kwargs = Python.dict()
    processor_kwargs["min_pixels"] = MIN_PIXELS
    processor_kwargs["max_pixels"] = MAX_PIXELS
    
    let processor = transformers.AutoProcessor.from_pretrained(BASE_PROCESSOR_NAME, **processor_kwargs)
    processor.tokenizer = tokenizer
    return processor

def to_device(data: PythonObject, device: PythonObject, dtype: PythonObject) -> PythonObject:
    """Recursively cast data into the specified device, dtype."""
    let torch = Python.import_module("torch")
    let collections = Python.import_module("collections.abc")
    
    if torch.is_tensor(data):
        return data.to(device=device, dtype=dtype)
    elif Python.isinstance(data, collections.Mapping):
        let new_dict = Python.dict()
        for key in data:
            new_dict[key] = to_device(data[key], device, dtype)
        return new_dict
    elif Python.isinstance(data, collections.Sequence) and not Python.isinstance(data, Python.import_module("builtins").str):
        let new_list = Python.list()
        for elem in data:
            new_list.append(to_device(elem, device, dtype))
        return new_list
    else:
        return data
