Following are additional notes on attempted or otherwise incomplete work, which may prove useful for any future research.

### Patch-wise auto-encoder
- At the moment, the features are extracted using a CNN that has simultaneous/parallel access to the entire image, and with a supervised learning signal; this is
	- unsatisfying when later down the line the aim is to show recognition working from sequential samples, as well as in an unsupervised setting
	- limits the extension of the architecture to a real-world setting where inputs are genuinely from serial samples end-to-end (e.g. images taking by a robot)
	- limits the exploration of other interesting tasks for the architecture, such as translation invariance (discussed further below)
- To address this, the code-base now contains an early implementation of a patch-wise auto-encoder that operates on 7x7 pixel patches, but this requires additional development and testing

### Visualising representations of unions
- Currently, the decoder for visualising GridCellNet representations cannot handle unions of representations
- At the point at which the network has successfully performed inference, although the representation will be a sub-set of the target class (e.g. the union of learned “seven” representations), it may sometimes be the union of several representations (say 2-3 learned examples of sevens that closely match the input)
- The decoder was trained on single SDR-based representations to reconstruct the input images, and so given larger union representations, it does not perform well - they end up looking like patchy/partially incomplete images
- It may be possible to still reconstruct these by generating enough examples to form a training data-set for the decoder, and then evaluating them on held-out examples
	- If so, the result may be quite interesting, as it may be a way for GridCellNet to encode a representation that is closer to the ground-truth input than any one learned example

### Translation invariance
- In theory, GridCellNet should work very well for translation invariance, as its representation is in the reference frame of the object, and so e.g. given a particular feature input, it will predict the next features relative to this point, not the actual spatial location in the input image; this has not yet been demonstrated on pixel-level tranlsations due to the issue of equivariance/invariance
#### Ensuring equivariant/invariant input features
- Despite weight sharing and max-pooling, normal CNNs are not actually very good at translation invariance or equivariance (see e.g. MNIST-C from Mu et al 2020). Thus translating MNIST digits and feeding them through the original feature extraction process may not work
- Given larger translations in the input, the desire would be to have an equivariant front-end – i.e. a translation of the input features causes an equivalent translation of the output features
	- There has been some work in the CNN literature to make them more translation equivariant (see Zhang 2019, “Making Convolutional Networks Shift-Invariant Again”)
- Over smaller translations (i.e. a few pixels, and thus within the receptive field of a feature representation), the feature represenation should be invariant.

### Few-shot learning to predict features in a sequence
- While this work focused on classification, another interesting direction in the future would be to compare GridCellNet's rapid ability to predict the next feature in a sequence to e.g. an RNN