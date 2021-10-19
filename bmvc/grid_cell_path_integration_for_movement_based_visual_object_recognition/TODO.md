Following are additional notes on attempted or otherwise incomplete work, which may prove useful for any future research.

### Patch-wise auto-encoder
- At the moment, the features are extracted using a CNN that has simultaneous/parallel access to the entire image, and with a supervised learning signal; this is
	- unsatisfying when later down the line the aim is to show recognition working from sequential samples, as well as in an unsupervised setting
	- limits the extension of the architecture to a real-world setting where inputs are genuinely from serial samples end-to-end (e.g. images taking by a robot)
	- limits the exploration of other interesting tasks for the architecture, such as translation invariance (discussed further below)
- A natural solution would be to use an auto-encoder that receives patches of pixels, and generates useful features for downstream learning; I implemented this, and while it was working in principle, the accuracy of downstream classifiers (including the RNN/k-NN, and GridCellNet) was much lower
	- It’s unclear exactly what the issue was, but it may be that the task is too simple (i.e. encoding a pixel patch from MNIST with a high-dimensional hidden layer, even given the constraint of sparsity) – I implemented a basic denoising auto-encoder to make the task more difficult and thus the features possibly more useful, but from what I managed this didn’t make a significant difference
	- An alternative approach might be to use a form of contrastive learning instead

### Visualising representations of unions
- Currently, the decoder for visualising GridCellNet representations cannot handle unions of representations
- At the point at which the network has successfully performed inference, although the representation will be a sub-set of the target class (e.g. the union of learned “seven” representations), it may sometimes be the union of several representations (say 2-3 learned examples of sevens that closely match the input)
- The decoder was trained on single SDR-based representations to reconstruct the input images, and so given larger union representations, it does not perform well - they end up looking like patchy/partially incomplete images
- It may be possible to still reconstruct these by generating enough examples to form a training data-set for the decoder, and then evaluating them on held-out examples
	- If so, the result may be quite interesting, as it may be a way for GridCellNet to encode a representation that is closer to the ground-truth input than any one learned example

### Translation invariance
- In theory, GridCellNet should work very well for translation invariance, as its representation is in the reference frame of the object, and so e.g. given a particular feature input, it will predict the next features relative to this point, not the actual spatial location in the input image; this has not yet been demonstrated for the two reasons below
#### Using sensor position information during classification
- The main issue is that at the moment, the recognition algorithm uses some information about the position of the sensor in the reference frame of the object to constrain the comparisons needed for classification
	- Unless one (unfairly) assumes that this information is appropriately updated when an object is translated, then this will limit the ability of the system to handle translations
	- This constraint can be relaxed, but means that larger unions emerge in the network, and this can significantly impact accuracy; additional approaches such as e.g. using multiple columns may be necessary to address this
#### Ensuring equivariant input features
- Despite weight sharing and max-pooling, normal CNNs are not actually very good at translation invariance or equivariance (see e.g. MNIST-C from Mu et al 2020). Thus translating MNIST digits and feeding them through the original feature extraction process may not work
- Rather than an invariant front-end, the desire would be to have an equivariant front-end – i.e. a translation of the input features causes an equivalent translation of the output features
	- There has been some work in the CNN literature to make them more translation equivariant (see Zhang 2019, “Making Convolutional Networks Shift-Invariant Again”)
- An alternative solution might be to train the patch-wise encoder described above, although this may run into issues if the patches during training are always taken from an e.g. 4x4 grid – what then happens when the shift in the input digit does not correspond to a shift of an entire grid-space? Thus it would likely need to be trained on many sub-patches randomly taken across the image, rather than at discrete locations, and it’s unclear how well this would work given the challenges noted above about getting the feature representations to be useful
#### Other thoughts
- An alternative task which would be less realistic, but which would at least allow the question of translation invariance to be explored, would be to shift the features after extraction; unfortunately:
	- This is made difficult due to no clear way of handling empty space from a feature point of view (i.e. it’s not trivial to just extend the feature space to accommodate translations in arbitrary directions)
	- Assuming no empty space is added, and the features instead wrap around, the grid-cell activations would need to wrap around at the same point as the input features, which would imply that the grid-cells have some privileged knowledge about the input ahead of time, and this would limit the generalizability of the classifier

### Catastrophic forgetting
- GridCellNet should perform well regardless of whether particular classes are blocked or interleaved; demonstrating robustness to learning in this setting would be an interesting future direction.

### Few-shot learning to predict features in a sequence
- While this work focused on classification, another interesting direction in the future would be to compare GridCellNet's rapid ability to predict the next feature in a sequence to e.g. an RNN