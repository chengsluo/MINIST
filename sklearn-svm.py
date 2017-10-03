import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
# %matplotlib inline

debug_size=1000
labeled_images = pd.read_csv('train.csv')
images = labeled_images.iloc[0:debug_size,1:]
labels = labeled_images.iloc[0:debug_size,:1] # 4000 is the size of mini-data
train_images, test_images,train_labels, test_labels = \
    train_test_split(images,labels,test_size=0.2,train_size=0.8, random_state=0)

# we load it into a numpy array and reshape it
# so that it is two-dimensional (28x28 pixels)

i=20
img=train_images.iloc[i].as_matrix()
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
plt.show()
# A histogram of this image's pixel values shows the range.
plt.hist(train_images.iloc[i])
plt.show()

# first,we use sklearn.svm module to create a vector classifier.
clf=svm.SVC()
# Next,we pass our training images and labels to the classifier's fit method
clf.fit(train_images,train_labels.values.ravel())
# Finally, the test images and labels are passed to the score method to see how well we trained our model. Fit will return a float between 0-1 indicating our accuracy on the test data set.
accuracy=clf.score(test_images,test_labels)
print ("first accuracy:",accuracy)


# improve this, just need simplify our images
test_images[test_images>0]=1
train_images[train_images>0]=1

img=train_images.iloc[i].as_matrix().reshape((28,28))
plt.imshow(img,cmap='binary')
plt.title(train_labels.iloc[i])
plt.show()

plt.hist(train_images.iloc[i])
plt.show()

# training again
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
accuracy=clf.score(test_images,test_labels)
print ("train accuracy:",accuracy)

test_data=pd.read_csv('test.csv')
test_data[test_data>0]=1
results=clf.predict(test_data[0:debug_size])

# print ("test result:",results)
# save labels

df = pd.DataFrame(results)
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True, index_label='ImageId')

