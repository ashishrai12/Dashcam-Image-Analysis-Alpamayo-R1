@value
struct ModelConfig:
    var model_type: String
    var num_history_steps: Int
    var num_future_steps: Int
    var time_step: Float64
    var use_quantization: Bool
    
    fn __init__(
        inout self,
        model_type: String = "alpamayo_r1",
        num_history_steps: Int = 16,
        num_future_steps: Int = 64,
        time_step: Float64 = 0.1,
        use_quantization: Bool = True
    ):
        self.model_type = model_type
        self.num_history_steps = num_history_steps
        self.num_future_steps = num_future_steps
        self.time_step = time_step
        self.use_quantization = use_quantization

    fn print_config(self):
        print("Model Configuration:")
        print(" - Type: " + self.model_type)
        print(" - Future Horizon: " + str(self.num_future_steps * self.time_step) + "s")
        print(" - Quantized: " + str(self.use_quantization))
