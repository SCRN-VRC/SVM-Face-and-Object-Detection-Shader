Shader "SVMFaceDetection/classify"
{
    Properties
    {
        _TexKernel ("Kernel Output", 2D) = "black" {}
        _TexAlphaInds ("Alpha + Indicies", 2D) = "black" {}
        _TexClassBuffer ("Classify Buffer", 2D) = "black" {}
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

            //RWStructuredBuffer<float4> buffer : register(u1);

            #define outRes float2(24., 6.)

            Texture2D<float4> _TexKernel;
            Texture2D<float4> _TexAlphaInds;
            Texture2D<float4> _TexClassBuffer;
            float4 _TexAlphaInds_TexelSize;
            float _Dst;
            float rho;

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
                o.uv.w = _TexAlphaInds.Load(uint3(0, 0, 0)).r; //rho
                return o;
            }

            float classify(uint px) {
                double total = -rho;
                // We use 1 pixel to store gamma and rho
                const uint MAX = round(_TexAlphaInds_TexelSize.z) - 1;
                for (uint i = 0; i < MAX; i++) {
                    uint index = round(_TexAlphaInds.Load(uint3(i, 1, 0)).r);
                    total += _TexAlphaInds.Load(uint3(i, 0, 0)).r * 
                        _TexKernel.Load(uint3(index,px,0)).r;
                }
                return total > 0 ? 0 : 1;
            }

            float4 frag (v2f ps) : SV_Target
            {
                clip(ps.uv.z);
                uint2 px = round(ps.uv.xy * outRes);
                rho = ps.uv.w;
                float4 col = _TexClassBuffer.Load(uint3(px, 0));
                if (col.g > 0.0) {
                    // Update rate, lower frame slower updates
                    col.g -= unity_DeltaTime.w * 0.003;
                }
                else {
                    col.r = classify(px.x + px.y * 24);
                    //Wavey pattern
                    col.g = 1.0 + 0.5*sin(ps.uv.x);
                }

                return col;
            }
            ENDCG
        }
    }
}
