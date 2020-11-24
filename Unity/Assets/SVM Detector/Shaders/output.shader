Shader "SVM Detector/output"
{
    Properties
    {
        _MainTex ("Camera", 2D) = "white" {}
        _Buffer ("Buffer", 2D) = "black" {}
        _Threshold ("Classify Threshold", Range(0, 1)) = 0.4
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

            sampler2D _MainTex;
            Texture2D<uint4> _Buffer;
            float4 _MainTex_TexelSize;
            float _Threshold;

            inline float box(float2 uv) {
                uv = step(0.95, uv);
                return max(uv.x, uv.y);
            }

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            fixed4 frag (v2f ps) : SV_Target
            {
                // sample the texture
                fixed4 col = tex2D(_MainTex, ps.uv);
                float2 p = ps.uv;

                uint i;
                for (i = 0; i < 27; i++) {
                    for (uint j = 0; j < 14; j++) {
                        uint2 pos = txPredict1.xy + uint2(j, i);
                        float cl = asfloat(_Buffer.Load(uint3(pos, 0)).r);
                        float2 uv2 = (float2(i, j) / float2(27, 14)) * float2(.9, 0.8) +
                            float2(0.07, 0.12);
                        float2 dist = p - uv2;
                        if (cl > _Threshold && abs(dist.x) < 0.06 && abs(dist.y) < 0.10667)
                        {
                            col.rgb *= float3(2, .5, .5);
                        }
                    }
                }

                for (i = 0; i < 17; i++) {
                    for (uint j = 0; j < 8; j++) {
                        uint2 pos = txPredict2.xy + uint2(j, i);
                        float cl = asfloat(_Buffer.Load(uint3(pos, 0)).r);
                        float2 uv2 = (float2(i, j) / float2(17, 8)) * float2(0.87, 0.7) + 
                            float2(0.14, 0.24);
                        float2 dist = p - uv2;
                        if (cl > _Threshold && abs(dist.x) < 0.12 && abs(dist.y) < 0.2133)
                        {
                            col.rgb *= float3(2, .5, .5);
                        }
                    }
                }

                for (i = 0; i < 8; i++) {
                    uint2 pos = txPredict3.xy + uint2(0, i);
                    float cl = asfloat(_Buffer.Load(uint3(pos, 0)).r);
                    float2 uv2 = float2(i, 1) / float2(8, 2);
                    uv2.x = uv2.x * 0.65 + 0.2;
                    float2 dist = p - uv2;
                    if (cl > _Threshold && abs(dist.x) < 0.24 && abs(dist.y) < 0.42667)
                    {
                        col.rgb *= float3(2, .5, .5);
                    }
                }

                col.rgb = saturate(col.rgb);
                return col;
            }
            ENDCG
        }
    }
}
