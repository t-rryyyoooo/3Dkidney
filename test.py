x = 0

x += (128 * 128 * 8 * 1)
x += (128 * 128 * 8 * 32 * 3)
x += (128 * 128 * 8 * 64 * 3)
x += (64 * 64 * 4 * 64 * 4)
x += (64 * 64 * 4 * 128 * 3)
x += (32 * 32 * 2 * 128 * 4)
x += (32 * 32 * 2 * 256 * 3)
x += (16 * 16 * 1 * 256 * 4)
x += (16 * 16 * 1 * 512 * 4)
x += (32 * 32 * 2 * 512)
x += (32 * 32 * 2 * 768)
x += (32 * 32 * 2 * 256 * 6)
x += (64 * 64 * 4 * 256)
x += (64 * 64 * 4 * 384)
x += (64 * 64 * 4 * 128 * 6)
x += (128 * 128 * 8 * 128)
x += (128 * 128 * 8 * 192)
x += (128 * 128 * 8 * 64 * 6)
x += (128 * 128 * 8 * 3)

parameters = 19077571
batchSize = 3
print("The number of neueons : ", x)
print("Total parameters : ", parameters)
print("Batch size : ", batchSize)
byte = (x * batchSize + parameters) * 8

print("{} byte".format(byte))
print("{} GB".format(byte * 10**(-9)))
