package divisio.djl.example;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;

public class NDArrayExample {
    public static void main(final String[] args) {
        // If you are training or running a model, the API will create the right NDManager for you.
        // If you just want to use the NDArray API for yourself, you need to start with your own
        // manager like this:
        final NDManager manager = NDManager.newBaseManager();
        // If available, this will give you GPU based calculation, if you want a specific GPU or the
        // CPU instead, you can use a different variant of the call to select a different device.
        final NDManager managerOnCPU = NDManager.newBaseManager(Device.cpu());

        // If you want to do calculations, you need to move your data into an NDArray. There
        // are various methods for it, the simplest ones wrap a single number or boolean:
        NDArray almostPi  = manager.create((float)Math.PI);
        NDArray almostE   = manager.create(Math.E);
        NDArray one       = manager.create((byte)1);
        NDArray theAnswer = manager.create(42);
        NDArray big       = manager.create(Long.MAX_VALUE);
        NDArray isTrue    = manager.create(true);
        // Printing these NDArrays shows us their properties. Apart from the values themselves,
        // three things are important about an NDArray:
        // - the data type
        // - the device they are on
        // - the Shape
        System.out.println(almostPi.getDataType()); //float32
        System.out.println(almostE.getDataType()); //float64
        System.out.println(one.getDataType()); //int8
        System.out.println(theAnswer.getDataType()); //int32
        System.out.println(big.getDataType()); //int64
        System.out.println(isTrue.getDataType()); //boolean
        // To get the data type of an NDArray, we can use NDArray.getDataType. Unlike Java data
        // structures like List, the NDArray does not contain its data type as a generic argument,
        // so there is no NDArray<Float>, just NDArray.
        // The type of the data stored depends on how the array
        // was created and can be one of FLOAT32, FLOAT64, FLOAT16, UINT8, INT32, INT8, INT64,
        // BOOLEAN or UNKNOWN. Each data type is represented by an enum instance in
        // ai.djl.ndarray.types.DataType. Some of these data types correspond to Java equivalents.
        // So when you create an NDArray with a Java primitive:
        // float -> float32
        // double -> float64
        // byte -> int8
        // int -> int32
        // long -> int64
        // boolean -> boolean
        // The only NDArray data types not corresponding directly to a java data type are UINT8 and
        // FLOAT16. To get these, you need to manually change the data type of an NDArray. Luckily,
        // this is easy:
        System.out.println(almostPi.toType(DataType.FLOAT16, true).getDataType());
        // The first argument of "toType" is the target data type, the second decides whether to
        // reuse the existing memory and overwrite the old array or whether to create a new one.
        // With copy = true a new NDArray is returned, with copy = false the old array is changed
        // in place and the array itself is returned.
        // prints uint8
        System.out.println(theAnswer.toType(DataType.UINT8, false).getDataType());
        // Now we can use one of the many mathematical functions of NDArray to perform
        // fast, optimized math operations:
        System.out.println(almostPi.sin().getFloat()); //-8.742278E-8
        // For a single value this of course does not make sense and does not improve speed,
        // actually, transferring data to the GPU, executing the operation and fetching it may
        // even be *slower* than just using java math. But if we create a lot of values, things get
        // interesting.
        /*
        float[] random = new float[1000 * 1000 * 100];
        Random rand = new Random();
        for (int i = 0; i < random.length; ++i) {
            random[i] = rand.nextFloat();
        }

        long start1 = System.currentTimeMillis();
        float[] sines1 = new float[random.length];
        for (int i = 0; i < random.length; ++i) {
            sines1[i] = (float)Math.sin(random[i]);
        }
        System.out.println(System.currentTimeMillis() - start1); //3298

        long start2 = System.currentTimeMillis();
        NDArray randOnGpu = manager.create(random);
        float[] sines2 = randOnGpu.sin().toFloatArray();
        System.out.println(System.currentTimeMillis() - start2);//536
        */
        // Even in this simple example, the GPU is already 6 times faster than the plan java code.
        // Actually, most calculations get a much larger boost. In this example the transfer
        // of the data by creating NDArray objects and getting the result back is actually what
        // takes most of the time so we do not benefit fully from the possible GPU speed.
        // As long as the data stays on the GPU, we are much, much faster.


        // We have now seen the most important feature of the NDArray - we can perform math
        // operations on many values at once without a loop. On hardware that supports it, like
        // GPUs, the calculations are executed in parallel, which results in a huge performance gain.

        // There are a number of additional ways to generate NDArrays, all of them can be found
        // either as members of NDManager of NDArray.
        // manager.create creates NDArrays out of single primitives, arrays of primitives or 2D
        // arrays of primitives.

        // manager.arange creates a "range" of numbers, e.g. 0,1,2,3 or 0.0, -0.1, -0.2, -0.3. The
        // start/end values and step can be controlled by parameters. This is often used for testing,
        // but can be necessary in real applications as well, e.g. to create offsets to indices etc.

        // manager.ones() creates an array full on ones, manager.zeros creates an array full of
        // zeros, randomNormal, randomUniform, randomMultinomial creates arrays filled with random
        // values. This is for example necessary to initialize neural networks, as they need to be
        // filled with random data before they start to learn.

        // Unless they end with "i", all methods on NDArrays that perform math operations create
        // a new NDArray with the respective result. Finally, NDArray.duplicate() allows copying
        // an existing NDArray.

        // - Shape
        // In the preceding example we saw how we can apply a math operation on every element in
        // an NDArray at once. The sinus function only takes one parameter, so it is easily applied
        // to every element. But what happens to methods that need more than one value, like addition?

        System.out.println(manager.create(2).add(manager.create(2)));
        System.out.println(manager.arange(0, 8).add(manager.arange(10, 18)));
        System.out.println(manager.arange(0, 8).add(manager.create(2)));
        System.out.println(manager.arange(0, 8).reshape(4, 2).add(manager.create(new int[]{100, 1000})));
    }
}
