Shader "SVMFaceDetection/kernel"
{
    Properties
    {
        _TexData ("Test Data", 2D) = "black" {}
        _TexSV ("Support Vectors", 2D) = "black" {}
        _TexAlphaInds ("Alpha + Indicies", 2D) = "black" {}
        _TexBuffer ("Kernel Buffer", 2D) = "black" {}
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

            #define outRes float2(237., 144.)

            Texture2D<float4> _TexData;
            Texture2D<float4> _TexSV;
            Texture2D<float4> _TexAlphaInds;
            Texture2D<float4> _TexBuffer;
            float4 _TexSV_TexelSize;
            float _Dst;
            float gamma;

			//RWStructuredBuffer<float4> buffer : register(u1);

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
                o.vertex = fixed4(v.uv * 2 - 1, 0, 1);
                #ifdef UNITY_UV_STARTS_AT_TOP
                v.uv.y = 1-v.uv.y;
                #endif
                o.uv.xy = UnityStereoTransformScreenSpaceTex(v.uv);
                o.uv.z = (distance(_WorldSpaceCameraPos,
                    mul(unity_ObjectToWorld, fixed4(0,0,0,1)).xyz) > _Dst) ? -1 : 1;
                o.uv.w = _TexAlphaInds.Load(uint3(0, 1, 0)).r; //gamma
                return o;
            }

            float kernel(uint2 px) {
                const uint MAX = round(_TexSV_TexelSize.z);

                double s = 0.0;
                // For each feature in the support vector
                for (uint l = 0; l < MAX; l++) {
                    float t0 = _TexSV.Load(uint3(l, px.x, 0)).r -
                        _TexData.Load(uint3(l, px.y, 0)).r;
                    s += t0 * t0;
                }
                return exp(s*-gamma);
            }

            float4 frag (v2f ps) : SV_Target
            {
                clip(ps.uv.z);
                uint2 px = round(ps.uv.xy * outRes);
                gamma = ps.uv.w;
                float4 col = _TexBuffer.Load(uint3(px, 0));
                if (col.g > 0.0) {
                    // Update rate, lower frame slower updates
                    col.g -= unity_DeltaTime.w * 0.003;
                }
                else {
                    col.r = kernel(px);
                    //Wavey pattern
                    col.g = 1.0 + 0.5*sin(ps.uv.y * 3.14);
                }

                return col;
            }
            ENDCG
        }
    }
}
