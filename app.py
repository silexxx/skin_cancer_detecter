import streamlit as st

from PIL import Image
image = st.sidebar.file_uploader("Upload")
a=image


count=0
if image != None:
	img = Image.open(image)
	count=1
	st.image(image,caption="Image",use_column_width=True)

if count==1:

	from fastai.vision import ImageDataBunch,get_transforms,imagenet_stats,cnn_learner,models,load_learner,open_image
	from fastai.metrics import error_rate
	import numpy as np

	bs = 32
	path = "classes"

	np.random.seed(42)
	data = ImageDataBunch.from_folder(path, train='.', valid_pct=0.2,ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)

	learn = cnn_learner(data, models.resnet34, metrics=error_rate).load("stage-1")
	learn.export()
	learn = load_learner("classes")


	cat, tensor, probs = learn.predict(open_image(a))


	l=list(probs)
	a=tensor.__str__()
	a=int(a.strip("tensor""()"))
	l=list(probs)[a]
	l=l.__str__()
	b=float(l.strip("tensor""()"))
	if b>=0.9:
		st.write("prediction :")	
		st.write(cat)
	else:
		st.write("prediction :")
		st.write("Not Sure")
