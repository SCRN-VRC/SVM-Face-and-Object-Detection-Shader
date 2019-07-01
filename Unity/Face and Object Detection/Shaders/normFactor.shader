Shader "FaceAndObjectDetect/normFactor"
{
    Properties
    {
        _TexOut ("normFactor", 2D) = "black" {}
        _TexBins ("Binning Output", 2D) = "black" {}
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

            #define outRes _TexOut_TexelSize.zw
            //RWStructuredBuffer<float4> buffer : register(u1);

            Texture2D<float4> _TexOut;
            Texture2D<float4> _TexBins;
            float4 _TexOut_TexelSize;
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
                16x16 Block Normalization 
                Takes the top left, top right, bottom left, bottom right 
                pixels and adds the data it contains all together.

                In:
                    px - bottom left pixel
                Out:
                    Sum of all the data in the four pixels
            */
            float blockNorm(uint2 px) {

                uint4 data1 = (_TexBins.Load(uint3(px, 0)).rgba);
                uint4 data2 = (_TexBins.Load(uint3(px + uint2(1, 0), 0)).rgba);
                uint4 data3 = (_TexBins.Load(uint3(px + uint2(0, 1), 0)).rgba);
                uint4 data4 = (_TexBins.Load(uint3(px + uint2(1, 1), 0)).rgba);
              
                //0x0000ffff
                uint mask = 65535;

                float4 ex0 = f16tof32(uint4(data1.x & mask, data1.x >> 16,
                                            data1.y & mask, data1.y >> 16));
                float4 ex1 = f16tof32(uint4(data1.z & mask, data1.z >> 16,
                                            data1.w & mask, data1.w >> 16));
                float4 ex2 = f16tof32(uint4(data2.x & mask, data2.x >> 16,
                                            data2.y & mask, data2.y >> 16));
                float4 ex3 = f16tof32(uint4(data2.z & mask, data2.z >> 16,
                                            data2.w & mask, data2.w >> 16));


                float4 ex4 = f16tof32(uint4(data3.x & mask, data3.x >> 16,
                                            data3.y & mask, data3.y >> 16));
                float4 ex5 = f16tof32(uint4(data3.z & mask, data3.z >> 16,
                                            data3.w & mask, data3.w >> 16));
                float4 ex6 = f16tof32(uint4(data4.x & mask, data4.x >> 16,
                                            data4.y & mask, data4.y >> 16));
                float4 ex7 = f16tof32(uint4(data4.z & mask, data4.z >> 16,
                                            data4.w & mask, data4.w >> 16));

                //if (any(ex0 < 0))
                //     buffer[0] = ex0;

                return ex0.x + ex0.y + ex0.z + ex0.w + ex1.x + ex1.y + ex1.z + ex1.w +
                       ex2.x + ex2.y + ex2.z + ex2.w + ex3.x + ex3.y + ex3.z + ex3.w +
                       ex4.x + ex4.y + ex4.z + ex4.w + ex5.x + ex5.y + ex5.z + ex5.w +
                       ex6.x + ex6.y + ex6.z + ex6.w + ex7.x + ex7.y + ex7.z + ex7.w;
            }

            float4 frag (v2f ps) : SV_Target
            {
                clip(ps.uv.z);
                uint2 px = round(ps.uv.xy * outRes);
                float4 col = float4(0.,0.,0.,0.);

                //160x90
                uint2 targetPx = px + floor(px / 7);
                col.rgba = blockNorm(targetPx).rrrr;

                return col;
            }
            ENDCG
        }
    }
}
