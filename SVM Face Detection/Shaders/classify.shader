Shader "SVMFaceDetection/classify"
{
    Properties
    {
        _TexBins ("Binning Output", 2D) = "black" {}
        _TexNorm ("Normalization Factor", 2D) = "black" {}
        _TexSV ("Support Vectors", 2D) = "black" {}
        _TexAlphaInds ("Alpha + Indicies", 2D) = "black" {}
        _TexTable ("LookUp Table", 2D) = "black" {}
    }
    SubShader
    {
        Tags { "Queue"="Overlay+1" "ForceNoShadowCasting"="True" "IgnoreProjector"="True" }

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

            RWStructuredBuffer<float4> buffer : register(u1);

            #define outRes float2(40., 37.)

            Texture2D<float4> _TexBins;
            Texture2D<float4> _TexNorm;
            Texture2D<float4> _TexSV;
            Texture2D<float4> _TexAlphaInds;
            Texture2D<float4> _TexTable;

            float4 _TexSV_TexelSize;

            float gamma;
            float rho;
            uint2 MAX_ITERS;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float4 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv.xy = v.uv;
                o.uv.z = _TexAlphaInds.Load(uint3(0, 0, 0)).r; //rho
                o.uv.w = _TexAlphaInds.Load(uint3(0, 1, 0)).r; //gamma
                return o;
            }

            inline float extract(uint4 idx, uint2 px) {

                px += uint2(6 - idx.y, idx.x);
                uint2 bpx = px + floor(px / 7);

                uint2 off = idx.w > 1 ?
                                idx.w > 2 ? 
                                    uint2(1, 0) :
                                    uint2(1, 1) :
                                idx.w > 0 ?
                                    uint2(0, 0) :
                                    uint2(0, 1);

                uint4 data = _TexBins.Load(uint3(bpx + off, 0)).rgba;

                //0x0000ffff
                uint mask = 65535;

                float number = f16tof32(
                    idx.z > 3 ?
                        idx.z > 5 ?
                            idx.z > 6 ?
                                data.w & mask :
                                data.w >> 16 :
                            idx.z > 4 ?
                                data.z & mask :
                                data.z >> 16 :
                        idx.z > 1 ?
                            idx.z > 2 ?
                                data.y & mask :
                                data.y >> 16 :
                            idx.z > 0 ?
                                data.x & mask :
                                data.x >> 16);

                return number / _TexNorm.Load(uint3(px, 0)).r;
            }

            double kernel(uint x, uint2 px) {
                double s = 0.0;
                //[unroll]
                for (uint i = 0; i < MAX_ITERS.x; i++) {
                    double t0 = _TexSV.Load(uint3(i, x, 0)).r -
                        extract(round(_TexTable.Load(uint3(i, 0, 0)).rgba),
                            px);
                    s += t0 * t0;
                }
                buffer[0] = double4(s, s*-gamma, -gamma, exp(s*-gamma));
                return exp(s*-gamma);

            }

            float classify(uint2 px) {

                double total = -rho;
                //[unroll(100)]
                for (uint i = 0; i < MAX_ITERS.y; i++) {
                    total += _TexAlphaInds.Load(uint3(i + 1, 1, 0)).r * 
                        kernel(round(_TexAlphaInds.Load(uint3(i + 1, 0, 0)).r), 
                            px);
                }
                return total > 0 ? 0 : 1.0;
            }

            float4 frag (v2f ps) : SV_Target
            {
                //if (fmod(_Time.y + ps.uv.y, 1) < 1) discard;

                uint2 px = round(ps.uv.xy * outRes);
                rho = ps.uv.z;
                gamma = ps.uv.w;
                MAX_ITERS = uint2(round(_TexSV_TexelSize.z),
                    round(_TexSV_TexelSize.w));

                float4 col = float4(0.,0.,0.,0.);

                if (px.y < 6) {
                    if (px.x >= 24) discard;
                    col = classify(uint2(2, 11) * 7);
                }
                else if (px.y < 18) {
                    if (px.x >= 31) discard;
                    //col = float4(0,0,0,1);
                }
                else {
                    //col = float4(0,0,0,1);
                }

                return col;
            }
            ENDCG
        }
    }
}
