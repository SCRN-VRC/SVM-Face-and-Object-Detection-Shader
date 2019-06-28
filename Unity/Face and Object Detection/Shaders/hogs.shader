Shader "SVMFaceDetection/hogs"
{
    Properties
    {
        _TexBins ("Binning Output", 2D) = "black" {}
        _TexNormFactor ("Normalization Factor", 2D) = "black" {}
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

            #define outRes float2(336., 84.)

            Texture2D<float4> _TexBins;
            Texture2D<float4> _TexNormFactor;
            float _Dst;

            //RWStructuredBuffer<float4> buffer : register(u1);

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

            inline uint is_odd(uint x) { return x & 0x1; }

            /*
                Normalize the magnitudes by every four pixel in bins layer

                In:
                    px - bottom left pixel
                    offset - The offset to the bottom left pixel of the 7x7 
                        block in _TexNormFactor
                Out:
                    Data divided by the norm factor and repacked
            */
            float4 hogs(uint2 px, uint2 offset) {

                uint2 sx = floor(px / 2) + offset.xy;
                uint2 bx = sx + uint2(is_odd(px.x), is_odd(px.y));
                uint4 data = _TexBins.Load(uint3(bx, 0)).rgba;
                //0x0000ffff
                uint mask = 65535;
                float4 ex0 = f16tof32(uint4(data.x & mask, data.x >> 16,
                                            data.y & mask, data.y >> 16));
                float4 ex1 = f16tof32(uint4(data.z & mask, data.z >> 16,
                                            data.w & mask, data.w >> 16));
                float factor = _TexNormFactor.Load(uint3(sx, 0)).r;
                ex0 /= factor;
                ex1 /= factor;
                uint4 buf1 = f32tof16(ex0);
                uint4 buf2 = f32tof16(ex1);
                buf1.x = buf1.x << 16; buf1.y = buf1.y << 16;
                buf1.z = buf1.z << 16; buf1.w = buf1.w << 16;
                buf1.x |= buf2.x; buf1.y |= buf2.y;
                buf1.z |= buf2.z; buf1.w |= buf2.w;
                return buf1;
            }

            float4 frag (v2f ps) : SV_Target
            {
                clip(ps.uv.z);
                uint2 px = round(ps.uv.xy * outRes);
                float4 col = hogs(px % 14,floor(px / 14) * 7);
                return col;
            }
            ENDCG
        }
    }
}
