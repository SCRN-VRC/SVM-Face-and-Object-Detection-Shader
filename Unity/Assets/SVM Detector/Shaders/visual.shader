Shader "SVM Detector/visual"
{
    Properties
    {
        _Buffer ("Buffer", 2D) = "black" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"
            #include "svmhelper.cginc"
            #pragma target 5.0

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            //RWStructuredBuffer<float4> buffer : register(u1);
            Texture2D<uint4> _Buffer;
            float4 _Buffer_TexelSize;
            float _Threshold;

            float2 rot2(float2 inCoords, float rot)
            {
                float sinRot;
                float cosRot;
                sincos(rot, sinRot, cosRot);
                return mul(float2x2(cosRot, -sinRot, sinRot, cosRot), inCoords);
            }

            // https://www.shadertoy.com/view/3tdSDj
            float udSegment( in float2 p, in float2 a, in float2 b )
            {
                float2 ba = b-a;
                float2 pa = p-a;
                float h = saturate( dot(pa,ba)/dot(ba,ba) );
                return length(pa-h*ba);
            }

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            float4 frag (v2f ps) : SV_Target
            {
                float4 scoreHOG[4] =
                { 
                    -1.0, 0., 0., 0.,
                    -1.0, 0., 0., 0.,
                    -1.0, 0., 0., 0.,
                    -1.0, 0., 0., 0.
                };

                // Find 4 highest scores
                uint i;
                for (i = 0; i < 8; i++) {
                    uint2 pos = txPredict3.xy + uint2(0, i);
                    float cl = asfloat(_Buffer.Load(uint3(pos, 0)).r);
                    // Only if current score is bigger than last entry
                    if (cl > scoreHOG[3].x)
                    {
                        float4 curHOG = float4(cl, 3, float2(0, i));
                        // Sorting
                        [unroll]
                        for (int j = 0; j < 4; j++) {
                            float4 prevHOG = curHOG.x > scoreHOG[j].x ? scoreHOG[j] : curHOG;
                            scoreHOG[j] = curHOG.x > scoreHOG[j].x ? curHOG : scoreHOG[j];
                            curHOG = curHOG.x > scoreHOG[j].x ? scoreHOG[j] : prevHOG;
                        }
                    }
                }

                for (i = 0; i < 17; i++) {
                    for (uint j = 0; j < 8; j++) {
                        uint2 pos = txPredict2.xy + uint2(j, i);
                        float cl = asfloat(_Buffer.Load(uint3(pos, 0)).r);
                        float4 curHOG = float4(cl, 2, float2(i, j));
                        // Only if current score is bigger than last entry
                        if (cl > scoreHOG[3].x)
                        {
                            // Sorting
                            [unroll]
                            for (int k = 0; k < 4; k++) {
                                float4 prevHOG = curHOG.x > scoreHOG[k].x ? scoreHOG[k] : curHOG;
                                scoreHOG[k] = curHOG.x > scoreHOG[k].x ? curHOG : scoreHOG[k];
                                curHOG = curHOG.x > scoreHOG[k].x ? scoreHOG[k] : prevHOG;
                            }
                        }
                    }
                }

                for (i = 0; i < 27; i++) {
                    for (uint j = 0; j < 14; j++) {
                        uint2 pos = txPredict1.xy + uint2(j, i);
                        float cl = asfloat(_Buffer.Load(uint3(pos, 0)).r);
                        float4 curHOG = float4(cl, 1, float2(i, j));
                        // Only if current score is bigger than last entry
                        if (cl > scoreHOG[3].x)
                        {
                            // Sorting
                            [unroll]
                            for (int k = 0; k < 4; k++) {
                                float4 prevHOG = curHOG.x > scoreHOG[k].x ? scoreHOG[k] : curHOG;
                                scoreHOG[k] = curHOG.x > scoreHOG[k].x ? curHOG : scoreHOG[k];
                                curHOG = curHOG.x > scoreHOG[k].x ? scoreHOG[k] : prevHOG;
                            }
                        }
                    }
                }

                float4 col = 1.0;
                float2 p = fmod(ps.uv * 2, 1.0);
                uint2 id = floor(ps.uv * 2);
                id.x = id.x + (1 - id.y) * 2;

                // get the camera input
                uint2 pos = (floor(scoreHOG[id.x].y) == 3) ? txCam3.xy :
                    (floor(scoreHOG[id.x].y) == 2) ? txCam2.xy : txCam1.xy;
                pos += scoreHOG[id.x].zw * 64 + p * 64;
                col.rgb = asfloat(_Buffer.Load(uint3(pos, 0)).r);

                // get the bins
                p = fmod(ps.uv * 16, 1.0);
                bool cam3 = floor(scoreHOG[id.x].y) == 3;
                pos = (cam3) ? txCam3Bin.xy :
                    (floor(scoreHOG[id.x].y) == 2) ? txCam2Bin.xy : txCam1Bin.xy;
                uint2 of2 = floor(fmod(ps.uv * 16, 8));
                pos += scoreHOG[id.x].wz * 8 + (cam3 ? of2 : of2.yx);

                float bins[8];
                uint4 buf = _Buffer.Load(uint3(pos, 0));
                bins[0] = f16tof32(buf.x >> 16);
                bins[1] = f16tof32(buf.x);
                bins[2] = f16tof32(buf.y >> 16);
                bins[3] = f16tof32(buf.y);
                bins[4] = f16tof32(buf.z >> 16);
                bins[5] = f16tof32(buf.z);
                bins[6] = f16tof32(buf.w >> 16);
                bins[7] = f16tof32(buf.w);

                // if (id.x == 0 && id.y == 0)
                // buffer[0] = float4(bins[0], bins[1],bins[2],bins[3]);

                float btop = 0.0;
                [unroll]
                for (i = 0; i < 7; i++) btop = max(btop, bins[i]);
                // squash the noise
                btop = btop < 0.5 ? 999999.0 : btop;

                const float ro = 0.5;

                // bin 0
                float l = (bins[0] / btop) * 0.4;
                float d = udSegment(p, float2(0.5 - l, 0.5), float2(0.5 + l, 0.5)) - 0.01;
                col.rgb = d < 0.01 ? float3(0.0, 1.0, 0.0) : col.rgb;

                // bin 1
                float2 pr = rot2(p - ro, 0.3927) + ro;
                l = (bins[1] / btop) * 0.4;
                d = udSegment(pr, float2(0.5 - l, 0.5), float2(0.5 + l, 0.5)) - 0.01;
                col.rgb = d < 0.01 ? float3(0.0, 1.0, 0.0) : col.rgb;

                // bin 2
                pr = rot2(p - ro, 0.7854) + ro;
                l = (bins[2] / btop) * 0.4;
                d = udSegment(pr, float2(0.5 - l, 0.5), float2(0.5 + l, 0.5)) - 0.01;
                col.rgb = d < 0.01 ? float3(0.0, 1.0, 0.0) : col.rgb;

                // bin 3
                pr = rot2(p - ro, 1.1781) + ro;
                l = (bins[3] / btop) * 0.4;
                d = udSegment(pr, float2(0.5 - l, 0.5), float2(0.5 + l, 0.5)) - 0.01;
                col.rgb = d < 0.01 ? float3(0.0, 1.0, 0.0) : col.rgb;

                // bin 4
                pr = rot2(p - ro, 1.5708) + ro;
                l = (bins[4] / btop) * 0.4;
                d = udSegment(pr, float2(0.5 - l, 0.5), float2(0.5 + l, 0.5)) - 0.01;
                col.rgb = d < 0.01 ? float3(0.0, 1.0, 0.0) : col.rgb;

                // bin 5
                pr = rot2(p - ro, 1.9635) + ro;
                l = (bins[5] / btop) * 0.4;
                d = udSegment(pr, float2(0.5 - l, 0.5), float2(0.5 + l, 0.5)) - 0.01;
                col.rgb = d < 0.01 ? float3(0.0, 1.0, 0.0) : col.rgb;

                // bin 6
                pr = rot2(p - ro, 2.3562) + ro;
                l = (bins[6] / btop) * 0.4;
                d = udSegment(pr, float2(0.5 - l, 0.5), float2(0.5 + l, 0.5)) - 0.01;
                col.rgb = d < 0.01 ? float3(0.0, 1.0, 0.0) : col.rgb;

                // bin 7
                pr = rot2(p - ro, 2.7489) + ro;
                l = (bins[7] / btop) * 0.4;
                d = udSegment(pr, float2(0.5 - l, 0.5), float2(0.5 + l, 0.5)) - 0.01;
                col.rgb = d < 0.01 ? float3(0.0, 1.0, 0.0) : col.rgb;

                float2 p1 = abs(fmod(ps.uv * 128 + 1.0, 8.0) - 1.0);
                float g1 = saturate(min(p1.x, p1.y) * 4);

                col.rgb = g1 > .9 ? col.rgb : float3(0.4, 0.4, 0.4);
                col = saturate(col);
                return col;
            }
            ENDCG
        }
    }
}
