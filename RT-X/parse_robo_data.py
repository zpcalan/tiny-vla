import pickle
import io
from PIL import Image
with open('/data/k8s/zpc/tiny-vla/RT-X/sample_berkeley_gnm_cory_hall.data.pickle', 'rb') as f:
# with open('/data/k8s/zpc/tiny-vla/RT-X/sample_000000087204.data.pickle', 'rb') as f:
    data = pickle.load(f)

print("single data keys are", data.keys())
print("step number is", len(data['steps']))
print("keys of single step", data['steps'][0].keys())
print("obs are", data['steps'][0]['observation'].keys())
import pdb; pdb.set_trace()
for idx, step in enumerate(data['steps']):
    for key, value in step.items():
        if key == 'observation':
            print("================================================")
            for k, v in value.items():
                if k == 'image':
                    image = Image.open(io.BytesIO(v))
                    # image.save(f"image_{idx}.jpg")
                else:
                    print("=====k: ", k)
                    print("=====v: ", v)
                    print("\n ")