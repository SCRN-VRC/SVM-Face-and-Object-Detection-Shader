#ifndef _SVM_HELPER
#define _SVM_HELPER

#define SV_NUM                   133

#define txCam1                   uint4(0, 0, 1728, 896)
#define txCam2                   uint4(0, 896, 1088, 512)
#define txCam3                   uint4(1088, 896, 64, 512)
#define txPredict1               uint4(2022, 0, 14, 27)
#define txPredict2               uint4(2036, 0, 8, 17)
#define txPredict3               uint4(2044, 0, 1, 8)

#define txCam1Bin                uint4(1840, 378, 112, 216)
#define txCam1Norm               uint4(1924, 0, 98, 189)
#define txCam1Hog                uint4(1728, 0, 196, 378)

#define txCam2Bin                uint4(1952, 378, 64, 136)
#define txCam2Norm               uint4(1938, 189, 56, 98)
#define txCam2Hog                uint4(1728, 378, 112, 238)

#define txCam3Bin                uint4(1994, 189, 8, 64)
#define txCam3Norm               uint4(2002, 189, 7, 56)
#define txCam3Hog                uint4(1924, 189, 14, 112)

#define txLC                     uint2(2046, 2047)
#define txTimer                  uint2(2047, 2047)

static const float fX[3][3] =
{
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};

static const float fY[3][3] =
{
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1
};

// float test(uint3 pos, uint maxNum)
// {
//     float r;
//     if (pos.z == 0)
//         r = (pos.x / 64.0) * (pos.y /  64.0) * 2.0 - 1.0;
//     else if (pos.z == 1)
//         r = ((64.0 - pos.x) / 64.0) * (pos.y /  64.0) * 2.0 - 1.0;
//     else
//         r = (pos.x / 64.0) * ((64.0 - pos.y) /  64.0) * 2.0 - 1.0;
//     r = pos.x > maxNum ? 0.0 : r;
//     r = pos.y > maxNum ? 0.0 : r;
//     return r;
// }

void getFeature(inout float hogs[8], Texture2D<uint4> tex, uint2 iniPos,
    uint2 offset, uint l)
{
    uint2 pos = offset * 14 + iniPos + uint2(l / 14, l % 14);
    uint4 buf = tex.Load(uint3(pos, 0));
    hogs[0] = f16tof32(buf[0] >> 16);
    hogs[1] = f16tof32(buf[0]);
    hogs[2] = f16tof32(buf[1] >> 16);
    hogs[3] = f16tof32(buf[1]);
    hogs[4] = f16tof32(buf[2] >> 16);
    hogs[5] = f16tof32(buf[2]);
    hogs[6] = f16tof32(buf[3] >> 16);
    hogs[7] = f16tof32(buf[3]);
}

float getSV(Texture2D<float> tex, uint k, uint m)
{
    return tex.Load(uint3((k % 20) * 40 + m % 40, (k / 20) * 40 + m / 40, 0)).x;
}

inline bool insideArea(in uint4 area, uint2 px)
{
    if (px.x >= area.x && px.x < (area.x + area.z) &&
        px.y >= area.y && px.y < (area.y + area.w))
    {
        return true;
    }
    return false;
}

inline uint4 LoadValueUint( in Texture2D<uint4> tex, in int2 re )
{
    return asuint(tex.Load(int3(re, 0)));
}

inline void StoreValueUint( in int2 txPos, in uint4 value, inout uint4 col,
    in int2 fragPos )
{
    col = all(fragPos == txPos) ? value : col;
}

inline float4 LoadValueFloat( in Texture2D<uint4> tex, in int2 re )
{
    return asfloat(tex.Load(int3(re, 0)));
}

inline void StoreValueFloat( in int2 txPos, in float4 value, inout uint4 col,
    in int2 fragPos )
{
    col = all(fragPos == txPos) ? asuint(value) : col;
}

#endif