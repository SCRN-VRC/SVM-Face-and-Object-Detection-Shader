Shader "SVM Detector/svm"
{
    Properties
    {
        _CamIn ("Cam Input", 2D) = "white" {}
        _Buffer ("Buffer In", 2D) = "black" {}
        _SV ("Support Vectors", 2D) = "black" {}
        _IdleTime ("Idle time (ms)", Range(5, 20)) = 10
        _MaxDist ("Max Distance", Float) = 0.1
    }
    SubShader
    {
        Tags { "Queue"="Overlay+1" "ForceNoShadowCasting"="True" "IgnoreProjector"="True" }
        ZWrite Off
        ZTest Always
        Cull Off
        
        Pass
        {
            Lighting Off
            SeparateSpecular Off
            Fog { Mode Off }
            
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 5.0

            #include "UnityCG.cginc"
            #include "svmhelper.cginc"

            //RWStructuredBuffer<float4> buffer : register(u1);
            sampler2D _CamIn;
            Texture2D<uint4> _Buffer;
            Texture2D<float> _SV;
            float4 _CamIn_TexelSize;
            float4 _Buffer_TexelSize;
            float _MaxDist;
            uint _IdleTime;

            // grid size
            static const float2 g_size = _Buffer_TexelSize.zw / 64.0;

            // sliding window 1
            static const float2 w1_size = 64.0 / _CamIn_TexelSize.zw;
            static const float2 w1_stride = 16.0 / _CamIn_TexelSize.zw;

            // sliding window 2
            static const float2 w2_size = 64.0 / (_CamIn_TexelSize.zw * 0.5);
            static const float2 w2_stride = 12.0 / (_CamIn_TexelSize.zw * 0.5);

            // sliding window 3
            static const float2 w3_size = 64.0 / (_CamIn_TexelSize.zw * 0.25);
            static const float2 w3_stride = 8.0 / (_CamIn_TexelSize.zw * 0.25);

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float3 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = float4(v.uv * 2 - 1, 0, 1);
                #ifdef UNITY_UV_STARTS_AT_TOP
                v.uv.y = 1-v.uv.y;
                #endif
                o.uv.xy = UnityStereoTransformScreenSpaceTex(v.uv);
                o.uv.z = (distance(_WorldSpaceCameraPos,
                    mul(unity_ObjectToWorld, float4(0,0,0,1)).xyz) > _MaxDist ||
                    !unity_OrthoParams.w) ?
                    -1 : 1;
                return o;
            }

            uint4 frag (v2f i) : SV_Target
            {
                clip(i.uv.z);
                uint2 px = _Buffer_TexelSize.zw * i.uv.xy;
                uint4 col = _Buffer.Load(uint3(px, 0));

                // 2ms per frame
                float timer = LoadValueFloat(_Buffer, txTimer);
                timer += unity_DeltaTime.x;

                //buffer[0].x = timer;

                if (timer < 0.003)
                {
                    StoreValueFloat(txTimer, timer, col, px);
                    return col;
                }
                else timer = 0.0;

                float lc = LoadValueFloat(_Buffer, txLC);
                uint lcFloor = floor(lc);

                //buffer[0].y = lc;

                // luminance
                if (lcFloor == 0 && insideArea(txCam1, px))
                {
                    px -= txCam1.xy;
                    float2 grid = fmod(i.uv.xy * g_size, 1.0);
                    // stride
                    uint2 id = floor(i.uv.xy * g_size);
                    // apply sliding window on 480 x 270
                    float3 camCol = tex2D(_CamIn, id * w1_stride + grid * w1_size);
                    col.x = asuint(0.2126 * camCol.r + 0.7152 * camCol.g + 0.0722 * camCol.b);
                    // col.x = asuint(0.2126 * test(uint3(grid * 64, 0), 64) +
                    //     0.7152 * test(uint3(grid * 64, 1), 64) +
                    //     0.0722 * test(uint3(grid * 64, 2), 64));
                }
                else if (lcFloor == 0 && insideArea(txCam2, px))
                {
                    px -= txCam2.xy;
                    float2 grid = fmod(i.uv.xy * g_size, 1.0);
                    // stride
                    uint2 id = floor((i.uv.xy - txCam2.xy / _Buffer_TexelSize.zw) * g_size);
                    // apply sliding window on 240 x 135
                    float3 camCol = tex2D(_CamIn, id * w2_stride + grid * w2_size);
                    col.x = asuint(0.2126 * camCol.r + 0.7152 * camCol.g + 0.0722 * camCol.b);
                }
                else if (lcFloor == 0 && insideArea(txCam3, px))
                {
                    px -= txCam3.xy;
                    float2 grid = fmod(i.uv.xy * g_size, 1.0);
                    // stride
                    uint2 id = floor((i.uv.xy - txCam3.xy / _Buffer_TexelSize.zw) * g_size);
                    // apply sliding window on 120 x 67
                    float3 camCol = tex2D(_CamIn, id.yx * w3_stride + grid * w3_size);
                    col.x = asuint(0.2126 * camCol.r + 0.7152 * camCol.g + 0.0722 * camCol.b);
                }
                // edge detect x
                else if (lcFloor == 1 && insideArea(txCam1, px))
                {
                    px -= txCam1.xy;
                    px %= 64;
                    uint2 id = floor(i.uv.xy * g_size);

                    // clamp it in the 64x64 square
                    int2 ofs = txCam1.xy + id * 64;
                    int x1 = ofs.x + px.x;
                    int x0 = clamp(x1 - 1, ofs.x, ofs.x + 63);
                    int x2 = clamp(x1 + 1, ofs.x, ofs.x + 63);
                    int y1 = ofs.y + px.y;
                    int y0 = clamp(y1 - 1, ofs.y, ofs.y + 63);
                    int y2 = clamp(y1 + 1, ofs.y, ofs.y + 63);

                    float _00 = asfloat(_Buffer.Load(uint3(x0, y0, 0)));
                    float _02 = asfloat(_Buffer.Load(uint3(x0, y2, 0)));
                    float _10 = asfloat(_Buffer.Load(uint3(x1, y0, 0)));
                    float _12 = asfloat(_Buffer.Load(uint3(x1, y2, 0)));
                    float _20 = asfloat(_Buffer.Load(uint3(x2, y0, 0)));
                    float _22 = asfloat(_Buffer.Load(uint3(x2, y2, 0)));
                    
                    // sobel
                    col.y = asuint(_00 * fX[0][0] + _02 * fX[0][2] +
                        _10 * fX[1][0] + _12 * fX[1][2] +
                        _20 * fX[2][0] + _22 * fX[2][2]);
                }
                else if (lcFloor == 1 && insideArea(txCam2, px))
                {
                    px -= txCam2.xy;
                    px %= 64;
                    uint2 id = floor((i.uv.xy - txCam2.xy / _Buffer_TexelSize.zw) * g_size);

                    // clamp it in the 64x64 square
                    int2 ofs = txCam2.xy + id * 64;
                    int x1 = ofs.x + px.x;
                    int x0 = clamp(x1 - 1, ofs.x, ofs.x + 63);
                    int x2 = clamp(x1 + 1, ofs.x, ofs.x + 63);
                    int y1 = ofs.y + px.y;
                    int y0 = clamp(y1 - 1, ofs.y, ofs.y + 63);
                    int y2 = clamp(y1 + 1, ofs.y, ofs.y + 63);

                    float _00 = asfloat(_Buffer.Load(uint3(x0, y0, 0)));
                    float _02 = asfloat(_Buffer.Load(uint3(x0, y2, 0)));
                    float _10 = asfloat(_Buffer.Load(uint3(x1, y0, 0)));
                    float _12 = asfloat(_Buffer.Load(uint3(x1, y2, 0)));
                    float _20 = asfloat(_Buffer.Load(uint3(x2, y0, 0)));
                    float _22 = asfloat(_Buffer.Load(uint3(x2, y2, 0)));
                    
                    // sobel
                    col.y = asuint(_00 * fX[0][0] + _02 * fX[0][2] +
                        _10 * fX[1][0] + _12 * fX[1][2] +
                        _20 * fX[2][0] + _22 * fX[2][2]);
                }
                else if (lcFloor == 1 && insideArea(txCam3, px))
                {
                    px -= txCam3.xy;
                    px %= 64;
                    uint2 id = floor((i.uv.xy - txCam3.xy / _Buffer_TexelSize.zw) * g_size);

                    // clamp it in the 64x64 square
                    int2 ofs = txCam3.xy + id.xy * 64;
                    int x1 = ofs.x + px.x;
                    int x0 = clamp(x1 - 1, ofs.x, ofs.x + 63);
                    int x2 = clamp(x1 + 1, ofs.x, ofs.x + 63);
                    int y1 = ofs.y + px.y;
                    int y0 = clamp(y1 - 1, ofs.y, ofs.y + 63);
                    int y2 = clamp(y1 + 1, ofs.y, ofs.y + 63);

                    float _00 = asfloat(_Buffer.Load(uint3(x0, y0, 0)));
                    float _02 = asfloat(_Buffer.Load(uint3(x0, y2, 0)));
                    float _10 = asfloat(_Buffer.Load(uint3(x1, y0, 0)));
                    float _12 = asfloat(_Buffer.Load(uint3(x1, y2, 0)));
                    float _20 = asfloat(_Buffer.Load(uint3(x2, y0, 0)));
                    float _22 = asfloat(_Buffer.Load(uint3(x2, y2, 0)));
                    
                    // sobel
                    col.y = asuint(_00 * fX[0][0] + _02 * fX[0][2] +
                        _10 * fX[1][0] + _12 * fX[1][2] +
                        _20 * fX[2][0] + _22 * fX[2][2]);
                }
                // edge detect y
                else if (lcFloor == 2 && insideArea(txCam1, px))
                {
                    px -= txCam1.xy;
                    px %= 64;
                    uint2 id = floor(i.uv.xy * g_size);

                    // clamp it in the 64x64 square
                    int2 ofs = txCam1.xy + id * 64;
                    int x1 = ofs.x + px.x;
                    int x0 = clamp(x1 - 1, ofs.x, ofs.x + 63);
                    int x2 = clamp(x1 + 1, ofs.x, ofs.x + 63);
                    int y1 = ofs.y + px.y;
                    int y0 = clamp(y1 - 1, ofs.y, ofs.y + 63);
                    int y2 = clamp(y1 + 1, ofs.y, ofs.y + 63);

                    float _00 = asfloat(_Buffer.Load(uint3(x0, y0, 0)));
                    float _01 = asfloat(_Buffer.Load(uint3(x0, y1, 0)));
                    float _02 = asfloat(_Buffer.Load(uint3(x0, y2, 0)));
                    float _20 = asfloat(_Buffer.Load(uint3(x2, y0, 0)));
                    float _21 = asfloat(_Buffer.Load(uint3(x2, y1, 0)));
                    float _22 = asfloat(_Buffer.Load(uint3(x2, y2, 0)));
                    
                    // sobel
                    col.z = asuint(_00 * fY[0][0] + _01 * fY[0][1] + _02 * fY[0][2] +
                        _20 * fY[2][0] + _21 * fY[2][1] + _22 * fY[2][2]);

                    // if (px.x == 0 && px.y == 0) {
                    //     buffer[0] = float4(ofs, x0, x2);
                    // }
                }
                else if (lcFloor == 2 && insideArea(txCam2, px))
                {
                    px -= txCam2.xy;
                    px %= 64;
                    uint2 id = floor((i.uv.xy - txCam2.xy / _Buffer_TexelSize.zw) * g_size);

                    // clamp it in the 64x64 square
                    int2 ofs = txCam2.xy + id * 64;
                    int x1 = ofs.x + px.x;
                    int x0 = clamp(x1 - 1, ofs.x, ofs.x + 63);
                    int x2 = clamp(x1 + 1, ofs.x, ofs.x + 63);
                    int y1 = ofs.y + px.y;
                    int y0 = clamp(y1 - 1, ofs.y, ofs.y + 63);
                    int y2 = clamp(y1 + 1, ofs.y, ofs.y + 63);

                    float _00 = asfloat(_Buffer.Load(uint3(x0, y0, 0)));
                    float _01 = asfloat(_Buffer.Load(uint3(x0, y1, 0)));
                    float _02 = asfloat(_Buffer.Load(uint3(x0, y2, 0)));
                    float _20 = asfloat(_Buffer.Load(uint3(x2, y0, 0)));
                    float _21 = asfloat(_Buffer.Load(uint3(x2, y1, 0)));
                    float _22 = asfloat(_Buffer.Load(uint3(x2, y2, 0)));
                    
                    // sobel
                    col.z = asuint(_00 * fY[0][0] + _01 * fY[0][1] + _02 * fY[0][2] +
                        _20 * fY[2][0] + _21 * fY[2][1] + _22 * fY[2][2]);
                }
                else if (lcFloor == 2 && insideArea(txCam3, px))
                {
                    px -= txCam3.xy;
                    px %= 64;
                    uint2 id = floor((i.uv.xy - txCam3.xy / _Buffer_TexelSize.zw) * g_size);

                    // clamp it in the 64x64 square
                    int2 ofs = txCam3.xy + id.xy * 64;
                    int x1 = ofs.x + px.x;
                    int x0 = clamp(x1 - 1, ofs.x, ofs.x + 63);
                    int x2 = clamp(x1 + 1, ofs.x, ofs.x + 63);
                    int y1 = ofs.y + px.y;
                    int y0 = clamp(y1 - 1, ofs.y, ofs.y + 63);
                    int y2 = clamp(y1 + 1, ofs.y, ofs.y + 63);

                    float _00 = asfloat(_Buffer.Load(uint3(x0, y0, 0)));
                    float _01 = asfloat(_Buffer.Load(uint3(x0, y1, 0)));
                    float _02 = asfloat(_Buffer.Load(uint3(x0, y2, 0)));
                    float _20 = asfloat(_Buffer.Load(uint3(x2, y0, 0)));
                    float _21 = asfloat(_Buffer.Load(uint3(x2, y1, 0)));
                    float _22 = asfloat(_Buffer.Load(uint3(x2, y2, 0)));
                    
                    // sobel
                    col.z = asuint(_00 * fY[0][0] + _01 * fY[0][1] + _02 * fY[0][2] +
                        _20 * fY[2][0] + _21 * fY[2][1] + _22 * fY[2][2]);
                }
                // magnitude and direction packed
                else if (lcFloor == 3 && insideArea(txCam1, px))
                {
                    px -= txCam1.xy;
                    float2 edges = asfloat(col.yz);
                    float mag = sqrt(pow(edges.x, 2) + pow(edges.y, 2));
                    float dir = atan2(edges.y, edges.x);
                    col.w = f32tof16(mag) << 16 | f32tof16(dir);
                    // if (px.x == 0 && px.y == 0) {
                    //     buffer[0] = (f16tof32(_Buffer.Load(uint3(0,0,0)).w));
                    //     //buffer[0] = float4(edges, mag, dir * 180.0 / UNITY_PI + (dir < 0. ? 180. : 0.));
                    // }
                }
                else if (lcFloor == 3 && insideArea(txCam2, px))
                {
                    px -= txCam2.xy;
                    float2 edges = asfloat(col.yz);
                    float mag = sqrt(pow(edges.x, 2) + pow(edges.y, 2));
                    float dir = atan2(edges.y, edges.x);
                    col.w = f32tof16(mag) << 16 | f32tof16(dir);
                }
                else if (lcFloor == 3 && insideArea(txCam3, px))
                {
                    px -= txCam3.xy;
                    float2 edges = asfloat(col.yz);
                    float mag = sqrt(pow(edges.x, 2) + pow(edges.y, 2));
                    float dir = atan2(edges.y, edges.x);
                    col.w = f32tof16(mag) << 16 | f32tof16(dir);
                }
                else if (lcFloor == 4 && insideArea(txCam1Bin, px))
                {
                    px -= txCam1Bin.xy;
                    uint i = (px.y / 8) * 64 + (px.y % 8) * 8;
                    uint j = (px.x / 8) * 64 + (px.x % 8) * 8;
                    float bins[8] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

                    for (uint x = 0; x < 8; x++) {
                        for (uint y = 0; y < 8; y++) {
                            uint ix = txCam1.x + i + x;
                            uint jy = txCam1.y + j + y;
                            uint buf = _Buffer.Load(uint3(ix, jy, 0)).w;
                            // radians to degrees (0, 180)
                            float dir = f16tof32(buf) * (180.0 / UNITY_PI);
                            dir += dir < 0.0 ? 180.0 : 0.0;
                            float mag = f16tof32(buf >> 16);
                            uint id = ((uint)floor(dir / 22.5f)) % 8;
                            // Last bin wraps around to bin 0, ratio of magnitude to next bin
                            float binR = dir > 157.5 ? (dir - 157.5) / 22.5 : 0.0;
                            bins[id] += mag * (1.0 - binR);
                            bins[(id + 1) % 8] += mag * binR;
                        }
                    }

                    col.x = f32tof16(bins[0]) << 16 | f32tof16(bins[1]);
                    col.y = f32tof16(bins[2]) << 16 | f32tof16(bins[3]);
                    col.z = f32tof16(bins[4]) << 16 | f32tof16(bins[5]);
                    col.w = f32tof16(bins[6]) << 16 | f32tof16(bins[7]);

                    // if (px.x == 8 && px.y == 0)
                    // {
                    //     buffer[0] = float4(bins[0], bins[1], bins[2], bins[3]);
                    // }
                }
                else if (lcFloor == 5 && insideArea(txCam1Norm, px))
                {
                    px -= txCam1Norm.xy;
                    uint i = txCam1Bin.x + (px.x / 7) * 8 + px.x % 7;
                    uint j = txCam1Bin.y + (px.y / 7) * 8 + px.y % 7;
                    
                    float s = 0.0;
                    uint4 buf1 = _Buffer.Load(uint3(i, j, 0));
                    uint4 buf2 = _Buffer.Load(uint3(i + 1, j, 0));
                    uint4 buf3 = _Buffer.Load(uint3(i, j + 1, 0));
                    uint4 buf4 = _Buffer.Load(uint3(i + 1, j + 1, 0));

                    for (uint x = 0; x < 4; x++) {
                        s += f16tof32(buf1[x]) + f16tof32(buf1[x] >> 16);
                        s += f16tof32(buf2[x]) + f16tof32(buf2[x] >> 16);
                        s += f16tof32(buf3[x]) + f16tof32(buf3[x] >> 16);
                        s += f16tof32(buf4[x]) + f16tof32(buf4[x] >> 16);
                    }

                    s = s < NOISE_THRESH ? 999999.0 : s;
                    col.r = asuint(s);

                    // if (px.x == 6 && px.y == 6)
                    // {
                    //     buffer[0] = s;
                    // }
                }
                else if (lcFloor == 6 && insideArea(txCam1Hog, px))
                {
                    px -= txCam1Hog.xy;
                    uint i = px.y % 14;
                    uint j = px.x % 14;
                    uint si = (i / 2);
                    uint sj = (j / 2);
                    uint bi = si + (i % 2);
                    uint bj = sj + (j % 2);
                    uint2 ofs = txCam1Norm.xy + (px / 14) * 7;
                    uint2 ofb = txCam1Bin.xy + (px / 14) * 8;

                    float s = asfloat(_Buffer.Load(uint3(uint2(si, sj) + ofs, 0)).x);
                    uint4 b = _Buffer.Load(uint3(uint2(bi, bj) + ofb, 0));

                    [unroll]
                    for (uint x = 0; x < 4; x++) {
                        float fa = f16tof32(b[x] >> 16) / s;
                        float fb = f16tof32(b[x]) / s;
                        col[x] = f32tof16(fa) << 16 | f32tof16(fb);
                    }

                    // if (px.x == 15 && px.y == 23)
                    // {
                    //     buffer[0] = float4(f16tof32(col.z >> 16), f16tof32(col.z),
                    //         f16tof32(col.w >> 16), f16tof32(col.w));
                    // }
                }
                else if (lcFloor == 4 && insideArea(txCam2Bin, px))
                {
                    px -= txCam2Bin.xy;
                    uint i = (px.y / 8) * 64 + (px.y % 8) * 8;
                    uint j = (px.x / 8) * 64 + (px.x % 8) * 8;
                    float bins[8] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

                    for (uint x = 0; x < 8; x++) {
                        for (uint y = 0; y < 8; y++) {
                            uint ix = txCam2.x + i + x;
                            uint jy = txCam2.y + j + y;
                            uint buf = _Buffer.Load(uint3(ix, jy, 0)).w;
                            // radians to degrees (0, 180)
                            float dir = f16tof32(buf) * (180.0 / UNITY_PI);
                            dir += dir < 0.0 ? 180.0 : 0.0;
                            float mag = f16tof32(buf >> 16);
                            uint id = ((uint)floor(dir / 22.5f)) % 8;
                            // Last bin wraps around to bin 0, ratio of magnitude to next bin
                            float binR = dir > 157.5 ? (dir - 157.5) / 22.5 : 0.0;
                            bins[id] += mag * (1.0 - binR);
                            bins[(id + 1) % 8] += mag * binR;
                        }
                    }

                    col.x = f32tof16(bins[0]) << 16 | f32tof16(bins[1]);
                    col.y = f32tof16(bins[2]) << 16 | f32tof16(bins[3]);
                    col.z = f32tof16(bins[4]) << 16 | f32tof16(bins[5]);
                    col.w = f32tof16(bins[6]) << 16 | f32tof16(bins[7]);
                }
                else if (lcFloor == 5 && insideArea(txCam2Norm, px))
                {
                    px -= txCam2Norm.xy;
                    uint i = txCam2Bin.x + (px.x / 7) * 8 + px.x % 7;
                    uint j = txCam2Bin.y + (px.y / 7) * 8 + px.y % 7;
                    
                    float s = 0.0;
                    uint4 buf1 = _Buffer.Load(uint3(i, j, 0));
                    uint4 buf2 = _Buffer.Load(uint3(i + 1, j, 0));
                    uint4 buf3 = _Buffer.Load(uint3(i, j + 1, 0));
                    uint4 buf4 = _Buffer.Load(uint3(i + 1, j + 1, 0));

                    for (uint x = 0; x < 4; x++) {
                        s += f16tof32(buf1[x]) + f16tof32(buf1[x] >> 16);
                        s += f16tof32(buf2[x]) + f16tof32(buf2[x] >> 16);
                        s += f16tof32(buf3[x]) + f16tof32(buf3[x] >> 16);
                        s += f16tof32(buf4[x]) + f16tof32(buf4[x] >> 16);
                    }

                    s = s < NOISE_THRESH ? 999999.0 : s;
                    col.r = asuint(s);
                }
                else if (lcFloor == 6 && insideArea(txCam2Hog, px))
                {
                    px -= txCam2Hog.xy;
                    uint i = px.y % 14;
                    uint j = px.x % 14;
                    uint si = (i / 2);
                    uint sj = (j / 2);
                    uint bi = si + (i % 2);
                    uint bj = sj + (j % 2);
                    uint2 ofs = txCam2Norm.xy + (px / 14) * 7;
                    uint2 ofb = txCam2Bin.xy + (px / 14) * 8;

                    float s = asfloat(_Buffer.Load(uint3(uint2(si, sj) + ofs, 0)).x);
                    uint4 b = _Buffer.Load(uint3(uint2(bi, bj) + ofb, 0));

                    [unroll]
                    for (uint x = 0; x < 4; x++) {
                        float fa = f16tof32(b[x] >> 16) / s;
                        float fb = f16tof32(b[x]) / s;
                        col[x] = f32tof16(fa) << 16 | f32tof16(fb);
                    }
                }
                else if (lcFloor == 4 && insideArea(txCam3Bin, px))
                {
                    px -= txCam3Bin.xy;
                    uint i = (px.x / 8) * 64 + (px.x % 8) * 8;
                    uint j = (px.y / 8) * 64 + (px.y % 8) * 8;
                    float bins[8] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

                    for (uint x = 0; x < 8; x++) {
                        for (uint y = 0; y < 8; y++) {
                            uint ix = txCam3.x + i + x;
                            uint jy = txCam3.y + j + y;
                            uint buf = _Buffer.Load(uint3(ix, jy, 0)).w;
                            // radians to degrees (0, 180)
                            float dir = f16tof32(buf) * (180.0 / UNITY_PI);
                            dir += dir < 0.0 ? 180.0 : 0.0;
                            float mag = f16tof32(buf >> 16);
                            uint id = ((uint)floor(dir / 22.5f)) % 8;
                            // Last bin wraps around to bin 0, ratio of magnitude to next bin
                            float binR = dir > 157.5 ? (dir - 157.5) / 22.5 : 0.0;
                            bins[id] += mag * (1.0 - binR);
                            bins[(id + 1) % 8] += mag * binR;
                        }
                    }

                    col.x = f32tof16(bins[0]) << 16 | f32tof16(bins[1]);
                    col.y = f32tof16(bins[2]) << 16 | f32tof16(bins[3]);
                    col.z = f32tof16(bins[4]) << 16 | f32tof16(bins[5]);
                    col.w = f32tof16(bins[6]) << 16 | f32tof16(bins[7]);
                }
                else if (lcFloor == 5 && insideArea(txCam3Norm, px))
                {
                    px -= txCam3Norm.xy;
                    uint i = txCam3Bin.x + (px.x / 7) * 8 + px.x % 7;
                    uint j = txCam3Bin.y + (px.y / 7) * 8 + px.y % 7;
                    
                    float s = 0.0;
                    uint4 buf1 = _Buffer.Load(uint3(i, j, 0));
                    uint4 buf2 = _Buffer.Load(uint3(i + 1, j, 0));
                    uint4 buf3 = _Buffer.Load(uint3(i, j + 1, 0));
                    uint4 buf4 = _Buffer.Load(uint3(i + 1, j + 1, 0));

                    for (uint x = 0; x < 4; x++) {
                        s += f16tof32(buf1[x]) + f16tof32(buf1[x] >> 16);
                        s += f16tof32(buf2[x]) + f16tof32(buf2[x] >> 16);
                        s += f16tof32(buf3[x]) + f16tof32(buf3[x] >> 16);
                        s += f16tof32(buf4[x]) + f16tof32(buf4[x] >> 16);
                    }

                    s = s < NOISE_THRESH ? 999999.0 : s;
                    col.r = asuint(s);
                }
                else if (lcFloor == 6 && insideArea(txCam3Hog, px))
                {
                    px -= txCam3Hog.xy;
                    uint i = px.y % 14;
                    uint j = px.x % 14;
                    uint si = (i / 2);
                    uint sj = (j / 2);
                    uint bi = si + (i % 2);
                    uint bj = sj + (j % 2);
                    uint2 ofs = txCam3Norm.xy + (px / 14) * 7;
                    uint2 ofb = txCam3Bin.xy + (px / 14) * 8;

                    float s = asfloat(_Buffer.Load(uint3(uint2(si, sj) + ofs, 0)).x);
                    uint4 b = _Buffer.Load(uint3(uint2(bi, bj) + ofb, 0));

                    [unroll]
                    for (uint x = 0; x < 4; x++) {
                        float fa = f16tof32(b[x] >> 16) / s;
                        float fb = f16tof32(b[x]) / s;
                        col[x] = f32tof16(fa) << 16 | f32tof16(fb);
                    }
                }
                else if (lcFloor == 7 && insideArea(txUnroll1, px))
                {
                    px -= txUnroll1.xy;
                    uint i = px.x % 14;
                    uint j = px.x / 14;

                    float gamma = _SV.Load(uint3(798, 799, 0)).x;
                    float dist[8];
                    for (uint k = 0; k < 8; k++) {
                        for (uint l = 0; l < 196; l++) {
                            // Extract HOG features from input
                            float hogs[8];
                            getFeature(hogs, _Buffer, txCam1Hog.xy, uint2(i, px.y), l);
                            [unroll]
                            for (uint m = 0; m < 8; m++) {
                                float t0 = getSV(_SV, k + j * 8, l * 8 + m) - hogs[m];
                                dist[k] += t0 * t0;
                            }
                        }
                        dist[k] = exp(dist[k] * -gamma);
                    }

                    [unroll]
                    for (uint x = 0; x < 4; x++) {
                        col[x] = (f32tof16(dist[x * 2]) << 16) | f32tof16(dist[x * 2 + 1]);
                    }
                }
                else if (lcFloor == 8 && insideArea(txUnroll2, px))
                {
                    px -= txUnroll2.xy;
                    uint i = px.x % 8;
                    uint j = px.x / 8;

                    float gamma = _SV.Load(uint3(798, 799, 0)).x;
                    float dist[8];
                    for (uint k = 0; k < 8; k++) {
                        dist[k] = 0.0;
                        for (uint l = 0; l < 196; l++) {
                            // Extract HOG features from input
                            float hogs[8];
                            getFeature(hogs, _Buffer, txCam2Hog.xy, uint2(i, px.y), l);
                            [unroll]
                            for (uint m = 0; m < 8; m++) {
                                float t0 = getSV(_SV, k + j * 8, l * 8 + m) - hogs[m];
                                dist[k] += t0 * t0;
                            }
                        }
                        dist[k] = exp(dist[k] * -gamma);
                    }

                    [unroll]
                    for (uint x = 0; x < 4; x++) {
                        col[x] = f32tof16(dist[x * 2]) << 16 | f32tof16(dist[x * 2 + 1]);
                    }
                }
                else if (lcFloor == 9 && insideArea(txUnroll3, px))
                {
                    px -= txUnroll3.xy;
                    uint j = px.x;
                    float gamma = _SV.Load(uint3(798, 799, 0)).x;
                    float dist[8];
                    for (uint k = 0; k < 8; k++) {
                        dist[k] = 0.0;
                        for (uint l = 0; l < 196; l++) {
                            // Extract HOG features from input
                            float hogs[8];
                            getFeature(hogs, _Buffer, txCam3Hog.xy, uint2(0, px.y), l);
                            [unroll]
                            for (uint m = 0; m < 8; m++) {
                                float t0 = getSV(_SV, k + j * 8, l * 8 + m) - hogs[m];
                                dist[k] += t0 * t0;
                            }
                        }
                        dist[k] = exp(dist[k] * -gamma);
                    }

                    [unroll]
                    for (uint x = 0; x < 4; x++) {
                        col[x] = f32tof16(dist[x * 2]) << 16 | f32tof16(dist[x * 2 + 1]);
                    }
                }
                else if (lcFloor == 10 && insideArea(txPredict1, px))
                {
                    px -= txPredict1.xy;
                    uint vidx[SV_NUM];
                    float dist[SV_NUM];
                    for (uint k = 0; k < SV_NUM; k++) {
                        vidx[k] = (uint)floor(_SV.Load(uint3(k, 798, 0)).x);
                        // index
                        uint l = (k % 8) / 2;
                        // shift value
                        uint s = (1 - ((k % 8) % 2)) * 16;
                        uint2 pos = txUnroll1.xy + px + (k / 8) * uint2(14, 0);
                        // scoop up the unrolled values
                        if (l == 0)
                            dist[k] = f16tof32(_Buffer.Load(uint3(pos, 0))[0] >> s);
                        else if (l == 1)
                            dist[k] = f16tof32(_Buffer.Load(uint3(pos, 0))[1] >> s);
                        else if (l == 2)
                            dist[k] = f16tof32(_Buffer.Load(uint3(pos, 0))[2] >> s);
                        else
                            dist[k] = f16tof32(_Buffer.Load(uint3(pos, 0))[3] >> s);
                    }

                    float rho = _SV.Load(uint3(799, 799, 0)).x;
                    float s = -rho;
                    for (uint n = 0; n < SV_NUM; n++) {
                        s += _SV.Load(uint3(n, 799, 0)).r * dist[vidx[n]];
                    }
                    col.r = asuint(s);
                }
                else if (lcFloor == 10 && insideArea(txPredict2, px))
                {
                    px -= txPredict2.xy;
                    uint vidx[SV_NUM];
                    float dist[SV_NUM];
                    for (uint k = 0; k < SV_NUM; k++) {
                        vidx[k] = (uint)floor(_SV.Load(uint3(k, 798, 0)).x);
                        // index
                        uint l = (k % 8) / 2;
                        // shift value
                        uint s = (1 - ((k % 8) % 2)) * 16;
                        uint2 pos = txUnroll2.xy + px + (k / 8) * uint2(8, 0);
                        // scoop up the unrolled values
                        if (l == 0)
                            dist[k] = f16tof32(_Buffer.Load(uint3(pos, 0))[0] >> s);
                        else if (l == 1)
                            dist[k] = f16tof32(_Buffer.Load(uint3(pos, 0))[1] >> s);
                        else if (l == 2)
                            dist[k] = f16tof32(_Buffer.Load(uint3(pos, 0))[2] >> s);
                        else
                            dist[k] = f16tof32(_Buffer.Load(uint3(pos, 0))[3] >> s);
                    }

                    float rho = _SV.Load(uint3(799, 799, 0)).x;
                    float s = -rho;
                    for (uint n = 0; n < SV_NUM; n++) {
                        s += _SV.Load(uint3(n, 799, 0)).r * dist[vidx[n]];
                    }
                    col.r = asuint(s);
                }
                else if (lcFloor == 10 && insideArea(txPredict3, px))
                {
                    px -= txPredict3.xy;
                    uint vidx[SV_NUM];
                    float dist[SV_NUM];
                    for (uint k = 0; k < SV_NUM; k++) {
                        vidx[k] = (uint)floor(_SV.Load(uint3(k, 798, 0)).x);
                        // index
                        uint l = (k % 8) / 2;
                        // shift value
                        uint s = (1 - ((k % 8) % 2)) * 16;
                        uint2 pos = txUnroll3.xy + px + (k / 8) * uint2(1, 0);
                        // scoop up the unrolled values
                        if (l == 0)
                            dist[k] = f16tof32(_Buffer.Load(uint3(pos, 0))[0] >> s);
                        else if (l == 1)
                            dist[k] = f16tof32(_Buffer.Load(uint3(pos, 0))[1] >> s);
                        else if (l == 2)
                            dist[k] = f16tof32(_Buffer.Load(uint3(pos, 0))[2] >> s);
                        else
                            dist[k] = f16tof32(_Buffer.Load(uint3(pos, 0))[3] >> s);
                    }

                    float rho = _SV.Load(uint3(799, 799, 0)).x;
                    float s = -rho;
                    for (uint n = 0; n < SV_NUM; n++) {
                        s += _SV.Load(uint3(n, 799, 0)).r * dist[vidx[n]];
                    }
                    col.r = asuint(s);
                }

                lc = fmod((lc + 1), 11 + _IdleTime);
                StoreValueFloat(txLC, lc, col, px);
                StoreValueFloat(txTimer, timer, col, px);
                return col;
            }
            ENDCG
        }
    }
}
