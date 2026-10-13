// Errors that cross the core and the lazy chunks (R6): both sides test instances of them.

export class BucketOverflowError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'BucketOverflowError';
  }
}
