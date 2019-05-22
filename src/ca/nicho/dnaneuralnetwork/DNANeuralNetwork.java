package ca.nicho.dnaneuralnetwork;

import java.util.ArrayList;

public class DNANeuralNetwork {

	public static final float WEIGHT_MUTATION_CONST = 0.1f;
	
	public final int size;
	public final int inputSize;
	public final int outputSize;
	public final int middleSize;
	public final int depth; // How many middle layers
	
	// The network adjacency matrix, normalized between -1 and 1 always
	public float[][] graph;
	public float[] cachedActivtations;
	
	public DNANeuralNetwork(int inputSize, int outputSize, int middleSize, int depth) {
		this.inputSize = inputSize;
		this.outputSize = outputSize;
		this.middleSize = middleSize;
		this.depth = depth;
		
		size = inputSize + outputSize + middleSize * depth;
		graph = new float[size][size];
	}
	
	public DNANeuralNetwork(DNANeuralNetwork p1, DNANeuralNetwork p2) {
		this(p1.inputSize, p1.outputSize, p1.middleSize, p1.depth);
		if(p1.size != p2.size) {
			throw new IllegalArgumentException("Parents must be the same size");
		}
		
		int v1 = (int)(Math.random() * size * size);
		int v2 = (int)(Math.random() * size * size);
		
		// Take parent 1 (start) section of network
		for(int i = 0; i < Math.min(v1, v2); i++) {
			int x = i % size;
			int y = i / size;
			graph[x][y] = p1.graph[x][y];
		}
		
		// Take parent 2 section of network
		for(int i = Math.min(v1, v2); i < Math.max(v1, v2); i++) {
			int x = i % size;
			int y = i / size;
			graph[x][y] = p2.graph[x][y];
		}
		
		// Take parent 1 (end) section of network
		for(int i = Math.max(v1, v2); i < size * size; i++) {
			int x = i % size;
			int y = i / size;
			graph[x][y] = p1.graph[x][y];
		}
		
	}
	
	/**
	 * Connection rules:
	 * 
	 * 1. First "size" nodes can only connect to indices >= inputSize
	 * 2. Output nodes may not connect to anything
	 * 3. Middle nodes may only connect to depths that are above itself
	 */
	public void connectNodeRandom() {
		
		// Starting node may only be in the scope of the input or middle layer
		int connectFrom = (int)(Math.random() * (inputSize + middleSize * depth));
		
		// If this is negative, the node is being formed from the input no a middle layer
		// otherwise it is a part of the middle layer, so we must find which layer it is.
		int startDepth = connectFrom - inputSize;
		if(startDepth < 0) {
			startDepth = -1;
		} else {
			startDepth = startDepth / middleSize;
		}
		
		// Get the minimum index from the start depth and input
		int minIndex = (startDepth + 1) * middleSize + inputSize;
		
		// End node may only be in the scope of the middle or output layer
		int connectTo = (int)(Math.random() * (size - minIndex)) + minIndex;
		
		graph[connectFrom][connectTo] = (float)Math.random() * 2 - 1;
	}
	
	/**
	 * Mutate any non 0 connection slightly
	 * 
	 * @param probability the chance of a mutation occuring
	 */
	public void mutateConnectionWeightRandom(float probability) {
		for(int i = 0; i < size; i++) {
			for(int o = 0; o < size; o++) {
				if(this.graph[i][o] != 0 && Math.random() < probability) {
					this.graph[i][o] += Math.random() * WEIGHT_MUTATION_CONST - WEIGHT_MUTATION_CONST / 2;
					if(this.graph[i][o] < -1) {
						this.graph[i][o] = -1;
					} else if (this.graph[i][o] > 1) {
						this.graph[i][o] = 1;
					}
				}
			}
		}
	}
	
	public float[] calculateOutput(float[] input) {
	
		if(input.length != inputSize) {
			throw new IllegalArgumentException("Size of input much match that of this networks");
		}
		
		// List of precalculated values. If null, that value must be computed
		float[] stored = new float[size];
	
		// Sum input layer. It is assumed the inputs are already normalized
		for(int i = 0; i < inputSize; i++) {
			stored[i] += input[i];
		}
		
		// Middle layer calculations
		for(int o = 0; o < depth; o++) {
			// Sum middle layers for this depth
			for(int i = 0; i < middleSize; i++) {
				int index = depth * o + i + inputSize;
				// Go through all possible connections for this node
				for(int j = 0; j < size; j++) {
					stored[index] += graph[j][index] * stored[j];
				}
			}
			// Normalize current layer with sigmoid
			for(int i = 0; i < middleSize; i++) {
				int index = depth * o + i + inputSize;
				stored[index] = sigmoid(stored[index]);
			}
		}
		
		// Sum output layer
		for(int i = 0; i < outputSize; i++) {
			int index = size - outputSize + i;
			for(int j = 0; j < size; j++) {
				stored[index] += graph[j][index] * stored[j];
			}
		}
		
		// Normalize output layer with sigmoid
		for(int i = 0; i < outputSize; i++) {
			int index = size - outputSize + i;
			stored[index] = sigmoid(stored[index]);
		}
		
		float[] output = new float[outputSize];
		for(int i = 0; i < outputSize; i++) {
			int index = size - outputSize + i;
			output[i] = stored[index];
		}
		
		this.cachedActivtations = stored;
		
		/*for(int i = 0; i < stored.length; i++) {
			System.out.print(stored[i] + " ");
		}
		System.out.println();*/
		
		return output;
		
	}
	
	public void print() {
		for(int i = 0; i < size; i++) {
			for(int o = 0; o < size; o++) {
				System.out.print(graph[i][o] + " ");
			}
			System.out.println();
		}
	}
	
	public void printOutput(float[] input) {
		float[] output = calculateOutput(input);
		for(int i = 0; i < output.length; i++) {
			System.out.print(output[i] + " ");
		}
		System.out.println();
	}

	
	public static float sigmoid(float x) {
		return  2 * (float)(1/( 1 + Math.pow(Math.E,(-1*x)))) - 1;
	}

}
