import pickle
import io
from PIL import Image
with open('/data/k8s/zpc/spirit-v1.5/home_proj/RT-X/sample_000000087204.data.pickle', 'rb') as f:
    data = pickle.load(f)

print(data.keys())
for key, value in data['steps'][0].items():
    print("=====key: ", key)
    if key == 'observation':
        for k, v in value.items():
            print("*****k: ", k)
            print("*****v: ", v)
            print("***********")
            if k == 'image':
                image = Image.open(io.BytesIO(v))
                image.save(f"image_{k}.jpg")
    else:
        print("=====value: ", value)
    print("================================================")