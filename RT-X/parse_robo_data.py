import pickle
import io
from PIL import Image
with open('/data/k8s/zpc/tiny-vla/RT-X/sample_berkeley_gnm_cory_hall.data.pickle', 'rb') as f:
    data = pickle.load(f)

import pdb; pdb.set_trace()
print(data.keys(), len(data['steps']))
for idx, step in enumerate(data['steps']):
    for key, value in step.items():
        if key == 'observation':
            print("================================================")
            for k, v in value.items():
                if k == 'image':
                    image = Image.open(io.BytesIO(v))
                    image.save(f"image_{idx}.jpg")
                else:
                    print("=====k: ", k)
                    print("=====v: ", v)
                    print("\n ")