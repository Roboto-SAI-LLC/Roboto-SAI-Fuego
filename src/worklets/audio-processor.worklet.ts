class AudioProcessor extends AudioWorkletProcessor {
  private readonly bufferSize: number;
  private buffer: Float32Array;
  private bufferIndex: number;

  constructor(options?: { processorOptions?: { bufferSize?: number } }) {
    super();
    const size = options?.processorOptions?.bufferSize;
    this.bufferSize = typeof size === 'number' && size > 0 ? size : 4096;
    this.buffer = new Float32Array(this.bufferSize);
    this.bufferIndex = 0;
  }

  process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const output = outputs[0];
    if (output && output[0]) {
      output[0].fill(0);
    }

    const input = inputs[0];
    if (!input || input.length === 0 || !input[0]) {
      return true;
    }

    const channel = input[0];
    let offset = 0;

    while (offset < channel.length) {
      const remaining = this.bufferSize - this.bufferIndex;
      const available = channel.length - offset;
      const toCopy = Math.min(remaining, available);

      this.buffer.set(channel.subarray(offset, offset + toCopy), this.bufferIndex);
      this.bufferIndex += toCopy;
      offset += toCopy;

      if (this.bufferIndex === this.bufferSize) {
        const chunk = this.buffer.slice(0, this.bufferSize);
        this.port.postMessage(chunk);
        this.bufferIndex = 0;
      }
    }

    return true;
  }
}

registerProcessor('audio-processor', AudioProcessor);
