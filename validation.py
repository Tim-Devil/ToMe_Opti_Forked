import timm
import tome

# Use any ViT model here (see timm.models.vision_transformer)
model_name = "vit_base_patch16_224"

# Load a pretrained model
model = timm.create_model(model_name, pretrained=True)

# Set this to be whatever device you want to benchmark on
# If you don't have a GPU, you can use "cpu" but you probably want to set the # runs to be lower
device = "cuda:0"
runs = 50
batch_size = 256  # Lower this if you don't have that much memory
input_size = model.default_cfg["input_size"]

# Baseline benchmark
baseline_throughput = tome.utils.benchmark(
    model,
    device=device,
    verbose=True,
    runs=runs,
    batch_size=batch_size,
    input_size=input_size
)

# Applying ToMe
# Simply patch the model after initialization to enable ToMe.

# Apply ToMe
tome.patch.timm(model)

# ToMe with r=16
model.r = 16
tome_throughput = tome.utils.benchmark(
    model,
    device=device,
    verbose=True,
    runs=runs,
    batch_size=batch_size,
    input_size=input_size
)
print(f"Throughput improvement: {tome_throughput / baseline_throughput:.2f}x")

# ToMe with r=16 and a decreasing schedule
model.r = (16, -1.0)
tome_decr_throughput = tome.utils.benchmark(
    model,
    device=device,
    verbose=True,
    runs=runs,
    batch_size=batch_size,
    input_size=input_size
)
print(f"Throughput improvement: {tome_decr_throughput / baseline_throughput:.2f}x")