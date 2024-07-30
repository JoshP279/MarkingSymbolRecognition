from MNISTModel import MNISTModel
import os

mnist_model = MNISTModel()

model_path = "mnist_model.h5"
if os.path.exists(model_path):
    mnist_model.load_model(model_path)
else:
    mnist_model.build_model()
    mnist_model.train_model()
    mnist_model.save_model(model_path)
    mnist_model.predict()

for x in range(1, 10):
    mnist_model.predict_custom_image(f"digits/digit{x}.png")