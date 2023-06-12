It's possible to create textures in CUDA for data access. It's also possible to write to an array from CUDA, then access it as a texture in OpenGL. I found [this link](https://stackoverflow.com/questions/20762828/crash-with-cuda-ogl-interop) helpful


## Setting up a CUDA texture for reading

1. define variables: first, the host data will be moved to a cuda array. The cuda array then will be mapped into a texture

```C
void *h_r_data; // host data
cudaArray *d_r_data; // cuda array 
cudaTextureObject_t h_texObj; // cuda texture
cudaExtent extent = make_cudaExtent(DIM,DIM,DIM); // data dim
```

2. create cuda texture

```C
//cudaArray Descriptor
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

// cuda array creation
cudaMalloc3DArray(&d_r_data, &channelDesc, extent, 0);

cudaMemcpy3DParms copyParams = {0};
copyParams.srcPtr = make_cudaPitchedPtr(h_r_data, extent.width * sizeof(float), extent.height, extent.depth);
copyParams.dstArray = d_r_data;
copyParams.extent = extent;
copyParams.kind = cudaMemcpyHostToDevice;
cudaMemcpy3D(&copyParams);


// cuda texture creation
cudaResourceDesc texRes;
memset(&texRes, 0, sizeof(cudaResourceDesc));
texRes.resType = cudaResourceTypeArray;
texRes.res.array.array = d_r_data;
cudaTextureDesc texDescr;
memset(&texDescr, 0, sizeof(cudaTextureDesc));
texDescr.normalizedCoords = false;
texDescr.filterMode = cudaFilterModeLinear;
texDescr.addressMode[0] = cudaAddressModeClamp;   // clamp
texDescr.addressMode[1] = cudaAddressModeClamp;
texDescr.addressMode[2] = cudaAddressModeClamp;
texDescr.readMode = cudaReadModeElementType;
cudaCreateTextureObject(&h_texObj, &texRes, &texDescr, NULL);
```

3. Send cuda texture to device

```C
// In CUDA file, define a device texture object
__constant__ cudaTextureObject_t d_texObj;

// In C, copy 
cudaMemcpyToSymbol(d_texObj, (void**) &(h_texObj), sizeof(cudaTextureObject_t),sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
```


## Setting up OpenGL and CUDA interoperability

1. define the OpenGL texture id and make cuda array and surface object to modify the array

```C
GLuint texture; // initialize texture in openGL
cudaArray_t viewCudaArray; // cuda array for writing

cudaGraphicsResource_t viewCudaResource;
cudaResourceDesc viewCudaArrayResourceDesc;

cudaSurfaceObject_t viewCudaSurfaceObject; // surf object to modify the array
```

2. OpenGL texture setup and connecting it to CUDA array

```C
void setupTexture() 
{
    // create texture normally
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, screen_width, screen_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
    // connect with cuda texture
    cudaGraphicsGLRegisterImage(&viewCudaResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard))
    memset(&viewCudaArrayResourceDesc, 0, sizeof(viewCudaArrayResourceDesc));
    viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
}
```

3. Render OpenGL texture every frame. The cuda kernel should receive the **CudaSurfaceObject** to fill the cuda array and NOT the cuda array itself

```C
void renderTexture()
{
    // ...
    // Get Cuda to understand OpenGL
    // upload CUDA results to texture
    checkCudaErrors(cudaGraphicsMapResources(1, &viewCudaResource));
    // get a cuda array representing mapped texture
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, viewCudaResource, 0, 0));
    viewCudaArrayResourceDesc.res.array.array = viewCudaArray;
    
    // object to write to the cuda array
    cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc))
    // your kernel
    call_kernel(gridSize, blockSize, viewCudaSurfaceObject);


    cudaDestroySurfaceObject(viewCudaSurfaceObject);
    cudaGraphicsUnmapResources(1, &viewCudaResource);
    cudaStreamSynchronize(0);
    // ...

}
```

## The CUDA part

1. send the cudaSurfaceObject to the kernel

```C
// kernel setup
void call_kernel(dim3 gridSize, dim3 blockSize, cudaSurfaceObject_t d_output) {
    d_kernel <<< gridSize, blockSize >>>(d_output);
}
```

2. this is how to access the texture in CUDA and how to write to the surface object: to access the texture, 

```C
__global__ void d_kernel(cudaSurfaceObject_t d_out)
{
    // indexing
    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    // ...

    // reading from cuda texture
    float val = tex3D<float>(d_texObj, i, j, k); // i/j/k from 0-1 like normal OpenGL texture access

    // ...

    // writing to cuda texture
    uint final_color = ....;
    surf2Dwrite(final_color, d_out, x * sizeof(unsigned int), y);

    // ...
}
```

