/*
    bins.shader
    This layer does the sliding window operation on the three camera inputs 
    and compresses the 64x64x1 pictures into 8x8x8 data structs.
*/

Shader "SVMFaceDetection/bins"
{

    Properties
    {
        _InputTex ("Input", 2D) = "black" {}
        _Dst ("Distance Clip", Float) = 0.05
    }

    SubShader
    {
        Tags { "Queue"="Overlay+1" "ForceNoShadowCasting"="True" "IgnoreProjector"="True" }
        ZWrite Off
        ZTest Always
        Cull Front

        Pass
        {
            Lighting Off
            SeparateSpecular Off
            Fog { Mode Off }

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma fragmentoption ARB_precision_hint_fastest
            #pragma target 5.0

            #include "UnityCG.cginc"

            #define outRes float2(192., 48.)

            //RWStructuredBuffer<float4> buffer : register(u1);
            
            Texture2D<float4> _InputTex;
            float _Dst;

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
                o.vertex = fixed4(v.uv * 2 - 1, 0, 1);
                #ifdef UNITY_UV_STARTS_AT_TOP
                v.uv.y = 1-v.uv.y;
                #endif
                o.uv.xy = UnityStereoTransformScreenSpaceTex(v.uv);
                o.uv.z = (distance(_WorldSpaceCameraPos,
                    mul(unity_ObjectToWorld, fixed4(0,0,0,1)).xyz) > _Dst) ? -1 : 1;
                return o;
            }
            
            /*
                Normally HOG descriptors have 9 bins, but to save
                computation I wanted to shove it all into one pixel 
                so I don't waste cycles doing the same thing.
                And the most data I can fit into ARGB Floats are
                2 halfs per channel in 4 channels.

                In:
                    px - starting pixel (bottom left)
                Out:
                    Magnitude binned into 8 halfs according to direction
            */
            float4 bin(uint2 px) {
                float bins[8];
                [unroll]
                for (uint i = 0; i < 8; i++) bins[i] = 0.0;
                uint2 c = uint2(0.,0.);
                [unroll]
                for (; c.x < 8; c++) {
                    [unroll]
                    for (; c.y < 8; c++) {
                        //.y = magnitude, .z = direction
                        float2 data = _InputTex.Load(uint3(px + c, 0)).yz;
                        uint binNo = ((uint)(floor(data.y / 22.5))) % 8;
                        float ratio = fmod(data.x, 22.5) / 22.5;
                        bins[binNo] += data.x * (1.0 - ratio);
                        bins[(binNo + 1) % 8] += data.x * ratio;
                    }
                }

                uint4 buf1 = f32tof16(float4(bins[0], bins[2], bins[4], bins[6]));
                uint4 buf2 = f32tof16(float4(bins[1], bins[3], bins[5], bins[7]));

                // Does buf1 = buf1 << 16 work?
                buf1.x = buf1.x << 16; buf1.y = buf1.y << 16;
                buf1.z = buf1.z << 16; buf1.w = buf1.w << 16;
                buf1.x |= buf2.x; buf1.y |= buf2.y;
                buf1.z |= buf2.z; buf1.w |= buf2.w;

                // if (px.x == 20 && px.y == 20 && fmod(_Time.y, 1) < 0.1) {
                //     float ex = f16tof32(buf1.x >> 16);
                //     buffer[0] = float4(ex, bins[0], 0, 0);
                // }

                return buf1;
            }

            float4 frag (v2f ps) : SV_Target
            {
                clip(ps.uv.z);
                uint2 px = round(ps.uv.xy * outRes);
                float4 col = float4(0.,0.,0.,0.);
                // 160x90, stride = 4
                uint2 targetPx = floor(px / 8) * 4 + (px % 8) * 8;
                col = bin(targetPx);
                return col;
            }
            ENDCG
        }
    }
}
