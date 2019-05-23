package ca.nicho.dnaneuralnetwork;

import java.nio.ByteBuffer;

public class DNANeuralNetwork implements Comparable<DNANeuralNetwork> {

	public static final float WEIGHT_MUTATION_CONST = 0.1f;
	
	public final int size;
	public final int inputSize;
	public final int outputSize;
	public final int middleSize;
	public final int depth; // How many middle layers
	
	// The network adjacency matrix, normalized between -1 and 1 always
	public float[][] graph;
	public float[] cachedActivtations;
	
	// Keep track of the fitness score
	public int score = 0;
		
	public DNANeuralNetwork(int inputSize, int outputSize, int middleSize, int depth) {
		this.inputSize = inputSize;
		this.outputSize = outputSize;
		this.middleSize = middleSize;
		this.depth = depth;
		
		size = inputSize + outputSize + middleSize * depth;
		graph = new float[size][size];
	}
	
	/**
	 * Create the network from the raw data
	 * @param data byte string containing all the data
	 */
	public DNANeuralNetwork(byte[] data) {
		ByteBuffer in = ByteBuffer.wrap(data);
		inputSize = in.getInt();
		outputSize = in.getInt();
		middleSize = in.getInt();
		depth = in.getInt();
		size = inputSize + outputSize + middleSize * depth;
		graph = new float[size][size];
				
		// Load the actual network data
		for(int i = 0; i < size; i++) {
			for(int o = 0; o < size; o++) {
				graph[i][o] = in.getFloat();
			}
		}
	}
	
	public DNANeuralNetwork(DNANeuralNetwork p1, DNANeuralNetwork p2) {
		this(p1.inputSize, p1.outputSize, p1.middleSize, p1.depth);
		if(p1.size != p2.size) {
			throw new IllegalArgumentException("Parents must be the same size");
		}
		
		// Take parent 1 (start) section of network
		for(int i = 0; i < size * size; i++) {
			
			int x = i % size;
			int y = i / size;
			graph[x][y] = Math.random() < 0.5 ? p1.graph[x][y] : p2.graph[x][y];
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
	public void breakConnectionRandom(float probability) {
		for(int i = 0; i < size; i++) {
			for(int o = 0; o < size; o++) {
				if(this.graph[i][o] != 0 && Math.random() < probability) {
					this.graph[i][o] = 0;
				}
			}
		}
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
	
	/**
	 * Forward propagate the input and calculate the value at each node
	 * @param input The input to send through the neural network
	 * @return The weighted output of the neural network
	 */
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
		
		// Set output values
		float[] output = new float[outputSize];
		for(int i = 0; i < outputSize; i++) {
			int index = size - outputSize + i;
			output[i] = stored[index];
		}
		
		// Cache the set of nodes and their values
		this.cachedActivtations = stored;
		
		return output;
	}
	
	/**
	 * Get the size of the network based on the amount of connections
	 */
	public int connectionCount() {
		int count = 0;
		for(int i = 0; i < size; i++) {
			for(int o = 0; o < size; o++) {
				if(this.graph[i][o] != 0) {
					count++;
				}
			}
		}
		return count;
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

	/**
	 * Special sigmoid to bound from -1 to 1
	 * @param x
	 * @return
	 */
	public static float sigmoid(float x) {
		return  2 * (float)(1/( 1 + Math.pow(Math.E,(-1*x)))) - 1;
	}

	@Override
	public int compareTo(DNANeuralNetwork o) {
		if(this.score < o.score) {
			return -1;
		} else if(this.score == o.score) {
			int c1 = this.connectionCount();
			int c2 = this.connectionCount();
			if(c1 < c2) {
				return -1;
			} else if(c1 == c2) {
				return 0;
			} else {
				return 1;
			}
		} else {
			return 1;
		}
	}
	
	@Override
	public String toString() {
		String sequence = "";
		int set = 0;
		byte value = 0b00;
		for(int i = 0; i < size; i++) {
			for(int o = 0; o < size; o++) {
				if(graph[i][o] != 0) {
					value += 1;
				}
				if(++set > 26) {
					sequence += (char)('A' + value);
					value = 0;
					set = 0;
				}
			}
		}
		return sequence;
	}
	
	/**
	 * Convert this network into a byte array which can be
	 * stored in a binary file
	 * @return
	 */
	public byte[] toBin() {
		// Determine how much space this network will take up
		int dataSize = 4 * 4; // 4 integer fields, each use 4 bytes to store
		dataSize += size * size * 4; // Total size of adjacency matric is size ^ 2, each float taking 4 bytes
				
		// Create buffer and put the basic values required to rebuild the structure
		ByteBuffer buffer = ByteBuffer.allocate(dataSize);
		buffer.putInt(inputSize);
		buffer.putInt(outputSize);
		buffer.putInt(middleSize);
		buffer.putInt(depth);

		// Put all the adjacency matrix values into the buffer
		for(int i = 0; i < size; i++) {
			for(int o = 0; o < size; o++) {
				buffer.putFloat(graph[i][o]);
			}
		}
		
		return buffer.array();
	}

}
