from miles.utils import ext_kernel_pins

# before any import of miles_ext_kernels, in every process that imports miles
ext_kernel_pins.install()
