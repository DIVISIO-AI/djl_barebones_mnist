package divisio.djl.example;

import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image.Flag;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.File;
import java.util.Arrays;
import java.util.List;

/**
 * This class turns the input of our classification (a file) into an Input for the neural net.
 * (It does the job of the Dataset, just for a single classification call), and it turns the output
 * of the classification into a readable probability distribution.
 */
public class MnistClassificationTranslator implements Translator<File, Classifications> {

    // the list of classes that can be detected by an MNIST classifier
    private final List<String> mnistClasses =
            Arrays.asList("0", "1", "2", "3", "4", "5", "6", "7", "8", "9");

    @Override
    public Classifications processOutput(final TranslatorContext ctx, final NDList list) {
        //The MNIST neural net creates a list of unnormalized probabilities for each possible class.
        //We first turn them into "real" probabilities (that sum to one) by using the softmax
        // function on the last dimension of the output
        final NDArray probabilities = list.singletonOrThrow().softmax(0);
        //Then we use the Classifications utility class with the possible class labels to get nicely
        // formatted output (otherwise we could just get the index of the highest probability with
        // the NDArray.argmax function)
        return new Classifications(mnistClasses, probabilities);
    }

    @Override
    public NDList processInput(final TranslatorContext ctx, final File input) throws Exception {
        final NDArray imageAsNdArray = ImageClassifierDataset.imageFileToNDArray(
                ctx.getNDManager(),
                input,
                Flag.GRAYSCALE)
                //We add a reshape operation, as our fully connected network expects the first
                // dimension to be a batch dimension - if we do not add a dummy batch dimension of
                // size 1, the input does not get flattened correctly.
                .reshape(1, 28, 28, 1);
        //This should normally not be done with real applications: It causes all data to be
        // extracted from the GPU memory and printed to stdout. We do it for testing purposes to
        // see if the data was loaded correctly and so we get an ASCII art preview of the image on
        // the command line
        debugPrintMnistNDArray(imageAsNdArray, 0);
        return new NDList(imageAsNdArray);
    }

    @Override
    public Batchifier getBatchifier() {
        return Batchifier.STACK;
    }

    /**
     * Helper method to check if the image was properly loaded.
     */
    public static void debugPrintMnistNDArray(final NDArray ndArray, final int batchIdx) {
        final float[] values = ndArray.toFloatArray();
        int offset = batchIdx * 28 * 28;
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                final float value = values[offset];
                final char c = value < 0.33f ? ' ' : (value < 0.66f ? 'x' : 'X');
                System.out.print(c);
                ++offset;
            }
            System.out.println();
        }
    }
}
