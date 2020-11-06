package divisio.djl.example;
import ai.djl.modality.cv.Image.Flag;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.util.Progress;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * {@link RandomAccessDataset} that loads images from a folder with subfolders, each image will be assigned the name
 * of the subfolder as label. Images are loaded with RGB colors or grayscale with the width and height of the image file.
 */
@SuppressWarnings("unused")
public class ImageClassifierDataset extends RandomAccessDataset {

    /**
     * Root folder containing the subfolders for each label
     */
    private final File rootFolder;

    /**
     * Whether to load the images in color or grayscale
     */
    private final Flag colorMode;

    /**
     * List of all image files in subfolders, listed at construction time so we know the amount of training data ahead
     * of time.
     */
    private final ArrayList<File> imageFiles;

    /**
     * List of all possible labels (the names of the subfolders), indices are class ids
     */
    private final ArrayList<String> labels;

    /**
     * Mapping from label name to class id
     */
    private final Map<String, Integer> labelToClassId;

    /**
     * Creates a new instance of {@link RandomAccessDataset} with the given configuration
     *
     * @param builder a builder with the necessary configurations
     */
    protected ImageClassifierDataset(final Builder builder) {
        super(builder);
        // Get the root folder
        this.rootFolder = builder.rootFolder;
        if (!rootFolder.isDirectory() || !rootFolder.canRead()) {
            throw new IllegalArgumentException("Cannot read from folder '" + rootFolder + "'.");
        }
        // load images in color or grayscale?
        this.colorMode = builder.colorMode;
        // init lists & lookups
        this.labelToClassId = new HashMap<>();
        this.labels = new ArrayList<>();
        this.imageFiles = new ArrayList<>();
        // loop over all subfolders
        final File[] children = this.rootFolder.listFiles();
        if (children == null) { throw new IllegalArgumentException("Folder '" + rootFolder + "' contains no subfolders."); }
        Arrays.sort(children);
        for (final File child : children) {
            // loop over all non-hidden subfolders
            if (child != null && child.isDirectory() && child.canRead() && !child.getName().startsWith(".")) {
                final String label = child.getName();
                labelToClassId.put(label, labels.size());
                labels.add(label);
                final File[] imageFileChildren = child.listFiles();
                // loop over all files in the subfolder
                if (imageFileChildren != null) {
                    Arrays.sort(imageFileChildren);
                    for (final File imageFileChild : imageFileChildren) {
                        if (imageFileChild.isFile() && imageFileChild.canRead()) {
                            imageFiles.add(imageFileChild);
                        }
                    }
                }
            }
        }
        imageFiles.trimToSize();
        labels.trimToSize();
    }

    /**
     * @return the root folder this dataset is loaded from
     */
    public File getRootFolder() {
        return rootFolder;
    }

    /**
     * @param label the readable label for the class
     * @return the id (0-based) of the label
     */
    public int getClassId(final String label) {
        return labelToClassId.get(label);
    }

    /**
     * @param classId 0-based id of the label
     * @return the readable label for the class
     */
    public String getLabel(final int classId) {
        return labels.get(classId);
    }

    /**
     * @return an immutable list of all possible labels
     */
    public List<String> getLabels() {
        return Collections.unmodifiableList(labels);
    }

    /**
     * Utility method to preprocess an image for usage in a neural net.
     * Turns the given image into an NDArray of shape [W,H,C]
     * and divides the values by 255 so they are in the range [0, 1].
     */
    public static NDArray imageFileToNDArray(final NDManager manager, final File file, final Flag colorMode) throws IOException {

        return ImageFactory.getInstance().fromFile(file.toPath()).toNDArray(manager, colorMode)
                //images are int values, neural networks need floats, so we need to transform the type
                // copy == false means that iff the array already had the correct type, no copy is created
                .toType(DataType.FLOAT32, false)
                //neural networks need normalized values, we change the range from 0-255 to 0-1
                // We do not need the old array, so we divide in-place with divi instead of div.
                .divi(255);
    }

    /**
     * Returns a record containing an image and its class id (index of the label).
     */
    @Override
    public Record get(final NDManager manager, final long index) throws IOException {
        final File imageFile = imageFiles.get((int)index);
        final String label = imageFile.getParentFile().getName();
        final int classId = labelToClassId.get(label);
        return new Record(
            new NDList(imageFileToNDArray(manager, imageFile, colorMode)),
            new NDList(manager.create(classId))
        );
    }

    /**
     * Returns the number of images that can be loaded by this Dataset. This is used to determine which indices are
     * valid when calling {@link ImageClassifierDataset#get(NDManager, long)}
     */
    @Override
    protected long availableSize() { return imageFiles.size(); }

    /**
     * Creates a new builder to build an {@link ImageClassifierDataset} from
     */
    public static Builder builder() { return new Builder(); }


    @Override
    public void prepare(Progress progress) {}

    /**
     * Builder for an {@link ImageClassifierDataset}.
     */
    public static class Builder extends RandomAccessDataset.BaseBuilder<ImageClassifierDataset.Builder> {

        protected File rootFolder;

        protected Flag colorMode = Flag.COLOR;

        protected Builder() {}

        /**
         * Sets the folder to read images from.
         * Folder must contain subdirectories that are named like the labels
         * and contain the images for each label.
         */
        public Builder setRootFolder(final File rootFolder) {
            this.rootFolder = rootFolder;
            return self();
        }

        /**
         * Returns the root folder currently set or throws an exception if none is set.
         */
        public File getRootFolder() {
            if (rootFolder == null) {
                throw new IllegalArgumentException("Root folder must be set.");
            }
            return rootFolder;
        }

        /**
         * Optionally sets a new color mode, the default is {@link Flag#COLOR}
         */
        public Builder optColorMode(final Flag colorMode) {
            this.colorMode = colorMode;
            return self();
        }

        /**
         * Returns the currently set color mode
         */
        public Flag getColorMode() {
            return colorMode;
        }

        @Override
        protected ImageClassifierDataset.Builder self() {
            return this;
        }

        public ImageClassifierDataset build() {
            return new ImageClassifierDataset(this);
        }
    }
}
