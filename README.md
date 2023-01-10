# **XOR Nerual Network** 

This is code is originally created by ***SamarAswani***

**source code:** https://github.com/SamarAswani/ANN-XOR

The code is divided into three part.

## **layer.c**
------------------------------------
Generating sigle layer, preset space, and some other uility.

|**function**|**descirption**|
|----|----|
|**double sigmoid(double x)**|Implementation of sigmoid function.|
|**double sigmoidprime(double x)**|Derivative of sigmoid function.|
|**layer_t \*layer_create()**|Preset space for layer.|
|**bool layer_init(layer_t \*layer, int num_outputs, layer_t \*prev)**|Initializing laryer, setup layer input, output, and preset memory space.|
|**void layer_free(layer_t \*layer)**|Free memeory space of single layer.|
|**void layer_compute_outputs(layer_t const \*layer)**|Using prevous layer out put as input and compute output for next layer.|
|**void layer_compute_deltas(layer_t const \*layer)**|Using next layer's deltas and weights to compute deltas for this layer|
|**void layer_update(layer_t const \*layer, double l_rate)**|Using computed deltas to update weight and bais.|
## **ann.c**
------------------------------------
Gererating, updating nerual network.

|**function**|**descirption**|
|----|----|
|**ann_t \*ann_create**|Generating neural network by given nodes and layers.|
|**void ann_free(ann_t \*ann)**|Free hold neural network|
|**void ann_predict(ann_t const \*ann, double const \*inputs)**|Computing output for whole neural network.|
|**void ann_train(ann_t const \*ann, double const \*inputs, double const \*targets, double l_rate)**|Training neural network|

## **teain.c**
------------------------------------

Training neural network using component from ann.c

## **original code**

```c
#include "ann.h"

/* Creates and trains a simple ann for XOR. */
int main()
{
  printf("Big data machine learning.\n\n");
  printf("--------------------------\n");

  /* Intializes random number generator */
  srand(42);

  /* Here is some BIG DATA to train, XOR function. */
  const double inputs[4][2] = {{0, 0},
                               {0, 1},
                               {1, 0},
                               {1, 1}};
  const double targets[] = {0, 1, 1, 0};

  printf("PART I - Creating a layer.\n\n");
  printf("Trying to layer_create.\n");
  layer_t *first_l = layer_create();
  if (!first_l) {
    printf("Couldn't create the first layer :(\n");
    return EXIT_FAILURE;
  }
  printf("Running layer_init.\n");
  if (layer_init(first_l, 2, NULL)) {
    printf("Couldn't layer_init first layer...\n");
    return EXIT_FAILURE;
  }
  printf("Here are some of the properties:\n");
  printf("  num_outputs: %i\n", first_l->num_outputs);
  printf("   num_inputs: %i\n", first_l->num_inputs);
  printf("   outputs[0]: %f\n", first_l->outputs[0]);
  printf("   outputs[1]: %f\n", first_l->outputs[1]);

  printf("\nCreating second layer.\n");
  layer_t *second_l = layer_create();
  if (!second_l) {
    printf("Couldn't create the second layer :(\n");
    return EXIT_FAILURE;
  }
  printf("Running layer_init on second layer.\n");
  if (layer_init(second_l, 1, first_l)) {
    printf("Couldn't layer_init second layer...\n");
    return EXIT_FAILURE;
  }
  printf("Here are some of the properties:\n");
  printf("  num_outputs: %i\n", second_l->num_outputs);
  printf("   num_inputs: %i\n", second_l->num_inputs);
  printf("   weights[0]: %f\n", second_l->weights[0][0]);
  printf("   weights[1]: %f\n", second_l->weights[1][0]);
  printf("    biases[0]: %f\n", second_l->biases[0]);
  printf("   outputs[0]: %f\n", second_l->outputs[0]);

  printf("\nComputing second layer outputs:\n");
  layer_compute_outputs(second_l);
  printf("Here is the new output:\n");
  printf("   outputs[0]: %f\n", second_l->outputs[0]);
  
  printf("\nFreeing both layers.\n");
  layer_free(second_l);
  layer_free(first_l);

  /* Create neural network. */
  printf("\n--------------------------\n");
  printf("PART II - Creating a neural network.\n");
  printf("2 inputs, 8 hidden neurons and 1 output.\n\n");
  printf(" * - * \\ \n");
  printf("         * - \n");
  printf(" * - * / \n\n");
  int layer_outputs[] = {2, 8, 1};
  ann_t *xor_ann = ann_create(3, layer_outputs);
  if (!xor_ann) {
    printf("Couldn't create the neural network :(\n");
    return EXIT_FAILURE;
  }

  /* Initialise weights to random. */
  printf("Initialising network with random weights...\n");

  /* Print hidden layer weights, biases and outputs. */
  printf("The current state of the hidden layer:\n");
  for(int i=0; i < layer_outputs[0]; ++i) {
    for(int j=0; j < layer_outputs[1]; ++j)
      printf("  weights[%i][%i]: %f\n", i, j, xor_ann->input_layer->next->weights[i][j]);
  }
  for(int i=0; i < layer_outputs[1]; ++i)
    printf("  biases[%i]: %f\n", i, xor_ann->input_layer->next->biases[i]);
  for(int i=0; i < layer_outputs[1]; ++i)
    printf("  outputs[%i]: %f\n", i, xor_ann->input_layer->next->outputs[i]);

  /* Dummy run to see random network output. */
  printf("Current random outputs of the network:\n");
  for(int i = 0; i < 4; ++i) {
    ann_predict(xor_ann, inputs[i]);
    printf("  [%1.f, %1.f] -> %f\n", inputs[i][0], inputs[i][1], xor_ann->output_layer->outputs[0]);
  }

  /* Train the network. */
  printf("\nTraining the network...\n");
  for(int i = 0; i < 25000; ++i) {
    /* This is an epoch, running through the entire data. */
    for(int j = 0; j < 4; ++j) {
      /* Training at batch size 1, ie updating weights after every data point. */
      ann_train(xor_ann, inputs[j], targets + j, 1.0);
    }
  }

  /* Print hidden layer weights, biases and outputs. */
  printf("The current state of the hidden layer:\n");
  for(int i=0; i < layer_outputs[0]; ++i) {
    for(int j=0; j < layer_outputs[1]; ++j)
      printf("  weights[%i][%i]: %f\n", i, j, xor_ann->input_layer->next->weights[i][j]);
  }
  for(int i=0; i < layer_outputs[1]; ++i)
    printf("  biases[%i]: %f\n", i, xor_ann->input_layer->next->biases[i]);
  for(int i=0; i < layer_outputs[1]; ++i)
    printf("  outputs[%i]: %f\n", i, xor_ann->input_layer->next->outputs[i]);

  /*Let's see the results. */
  printf("\nAfter training magic happened the outputs are:\n");
  for(int i = 0; i < 4; ++i) {
    ann_predict(xor_ann, inputs[i]);
    printf("  [%1.f, %1.f] -> %f\n", inputs[i][0], inputs[i][1], xor_ann->output_layer->outputs[0]);
  }

  /* Time to clean up. */
  ann_free(xor_ann);

  return EXIT_SUCCESS;
}
```
## **Output**
```sh
datastructure@ubuntu:~/Documents/ANN-XOR-main3$ ./train
Big data machine learning.

--------------------------
PART I - Creating a layer.

Trying to layer_create.
Running layer_init.
Here are some of the properties:
  num_outputs: 2
   num_inputs: 0
   outputs[0]: 0.000000
   outputs[1]: 0.000000

Creating second layer.
Running layer_init on second layer.
Here are some of the properties:
  num_outputs: 1
   num_inputs: 2
   weights[0]: -0.466530
   weights[1]: -0.170036
    biases[0]: 0.000000
   outputs[0]: 0.000000

Computing second layer outputs:
Here is the new output:
   outputs[0]: 0.500000

Freeing both layers.

--------------------------
PART II - Creating a neural network.
2 inputs, 8 hidden neurons and 1 output.

 * - * \ 
         * - 
 * - * / 

Initialising network with random weights...
The current state of the hidden layer:
  weights[0][0]: 0.190636
  weights[0][1]: -0.077513
  weights[0][2]: -0.293735
  weights[0][3]: -0.249872
  weights[0][4]: 0.136559
  weights[0][5]: 0.363622
  weights[0][6]: -0.198344
  weights[0][7]: -0.475076
  weights[1][0]: -0.135007
  weights[1][1]: 0.265381
  weights[1][2]: -0.182140
  weights[1][3]: -0.364272
  weights[1][4]: -0.393255
  weights[1][5]: 0.260672
  weights[1][6]: -0.418328
  weights[1][7]: 0.050956
  biases[0]: 0.000000
  biases[1]: 0.000000
  biases[2]: 0.000000
  biases[3]: 0.000000
  biases[4]: 0.000000
  biases[5]: 0.000000
  biases[6]: 0.000000
  biases[7]: 0.000000
  outputs[0]: 0.000000
  outputs[1]: 0.000000
  outputs[2]: 0.000000
  outputs[3]: 0.000000
  outputs[4]: 0.000000
  outputs[5]: 0.000000
  outputs[6]: 0.000000
  outputs[7]: 0.000000
Current random outputs of the network:
  [0, 0] -> 0.575744
  [0, 1] -> 0.579658
  [1, 0] -> 0.568582
  [1, 1] -> 0.572521

Training the network...
The current state of the hidden layer:
  weights[0][0]: 805.945472
  weights[0][1]: -682.349229
  weights[0][2]: -660.547427
  weights[0][3]: -607.748128
  weights[0][4]: 476.325807
  weights[0][5]: 743.268419
  weights[0][6]: -186.919007
  weights[0][7]: -696.847884
  weights[1][0]: -808.984264
  weights[1][1]: 679.236940
  weights[1][2]: 657.421482
  weights[1][3]: -607.764541
  weights[1][4]: -479.572707
  weights[1][5]: -746.344704
  weights[1][6]: -186.927165
  weights[1][7]: 693.744485
  biases[0]: 2.073698
  biases[1]: 2.174701
  biases[2]: 2.193165
  biases[3]: 606.070053
  biases[4]: 2.354378
  biases[5]: 2.125172
  biases[6]: 185.300611
  biases[7]: 2.162651
  outputs[0]: 0.290233
  outputs[1]: 0.293651
  outputs[2]: 0.294258
  outputs[3]: 0.000000
  outputs[4]: 0.299416
  outputs[5]: 0.291989
  outputs[6]: 0.000000
  outputs[7]: 0.293252

After training magic happened the outputs are:
  [0, 0] -> 0.002612
  [0, 1] -> 0.997584
  [1, 0] -> 0.997284
  [1, 1] -> 0.002324
```

Further analysis: [109503522_assignment.pdf)](./109503522_assignment.pdf)
