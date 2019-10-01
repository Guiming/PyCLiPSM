#from __future__ import absolute_import, print_function
import pyopencl as cl
import pkg_resources
print 'PyOpenCL version:', pkg_resources.get_distribution("pyopencl").version

#os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
print cl.device_type.CPU, cl.device_type.GPU

for platform in cl.get_platforms():
    print '====================='
    print 'Platform - Name:', platform.name
    print 'Platform - Vendor:', platform.vendor
    print 'Platform - Version:', platform.version
    print 'Platform - Profile:', platform.profile

    for device in platform.get_devices():
        print '\t-----------------------'
        print '\tDevice - Name:', device.name
        print '\tDevice - Type:', cl.device_type.to_string(device.type)
        print '\tDevice - Max Clock Speed:', device.max_clock_frequency, 'Mhz'
        print '\tDevice - Compute Units:', device.max_compute_units
        print '\tDevice - Local Memory:', device.local_mem_size / 1024.0, 'KB'
        print '\tDevice - Constant Memory:', device.max_constant_buffer_size / 1024.0, 'KB'
        print '\tDevice - Global Memory:', device.global_mem_size/1024**2, 'MB'
        print '\tDevice - Max Buffer/Image Size:', device.max_mem_alloc_size/1024**2, 'MB'
        print '\tDevice - Max Work Group Size:', device.max_work_group_size
        print '\tDevice - Max Work Item Dimensions:', device.max_work_item_dimensions
        print '\tDevice - Max Work Item Sizes:', device.max_work_item_sizes
        print '\n'
