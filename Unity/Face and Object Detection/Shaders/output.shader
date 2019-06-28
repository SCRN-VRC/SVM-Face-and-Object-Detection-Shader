Shader "SVMFaceDetection/output"
{
    Properties
    {
    	_TexCam ("High Res Camera", 2D) = "black" {}
        _TexClass ("Classify Output", 2D) = "black" {}
        //_TexOutline ("Outline", 2D) = "black" {}
        _Color ("Color", Color) = (0.,1.,0.,1)
        _Size ("Size", Float) = 0.15
    }
    SubShader
    {
        Cull Off
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma fragmentoption ARB_precision_hint_fastest
            #pragma target 5.0

            #include "UnityCG.cginc"

            Texture2D<float4> _TexCam;
            Texture2D<float4> _TexClass;
            //sampler2D _TexOutline;
            float4 _TexCam_TexelSize;
            float4 _TexClass_TexelSize;
            float4 _Color;
            float _Size;

            RWStructuredBuffer<float4> buffer : register(u1);

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

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            inline float box(float2 uv) {
                uv = step(0.95, uv);
                return max(uv.x, uv.y);
            }

            float4 frag (v2f ps) : SV_Target
            {
                float4 col = _TexCam.Load(uint3(ps.uv * _TexCam_TexelSize.zw, 0));
                float2 size = _Size.xx;
                size.y = size.y / _TexCam_TexelSize.w * _TexCam_TexelSize.z;
                // The edges of the classified output has a lot of false 
                // positives
                for (uint x = 1; x < (uint)(_TexClass_TexelSize.z - 1); x++) {
                    for (uint y = 1; y < (uint)(_TexClass_TexelSize.w - 1); y++) {
                        bool isFace = _TexClass.Load(uint3(x, y, 0)).r > 0.5 ? true : false;
                        float2 muv = float2(x, y) / _TexClass_TexelSize.zw;
                        float2 dist = ps.uv - muv;
                        if (isFace && abs(dist.x) < size.x && abs(dist.y) < size.y) {
                            col += box(abs(dist) / size) * _Color;
                        }
                    }
                }
                return col;
            }
            ENDCG
        }
    }
}
