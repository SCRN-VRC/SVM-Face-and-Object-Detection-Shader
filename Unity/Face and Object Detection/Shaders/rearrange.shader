Shader "FaceAndObjectDetect/rearrange"
{
    Properties
    {
        _TexOut ("testData", 2D) = "black" {}
        _TexHOGs ("HOG Features", 2D) = "black" {}
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

            Texture2D<float4> _TexOut;
            Texture2D<float4> _TexHOGs;
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

            inline uint is_odd(uint x) { return x & 0x1; }

            /*
                Merlin told me to put this on its own render texture
                to reduce the number conditional moves when unpacking 
                the float16. It works wonders.

                This stretches out the 14x14x8 render texture into 1568x1

                In:
                    px = Current pixel, we use it to determine one pixel of
                        one of the 14x14 block to grab
                    offset = Offset over to the correct 14x14 block
                Out:
                    The float from mapping 14x14x8 into 1568x1
            */

            float flatten(uint2 px, uint2 offset) {
                uint4 buffer;
                buffer.x = px.x % 14;
                // Flip the y cause OpenCV starts 0,0 at top left
                // And all the data is trained in OpenCV
                buffer.y = (((uint) floor((14 - px.x) / 14)) % 14);
                buffer.z = px.x % 8;
                buffer.w = _TexHOGs.Load(uint3(buffer.xy + offset.xy, 0))[buffer.z >> 1];
                //0x0000ffff
                uint mask = 65535;
                return f16tof32(is_odd(buffer.z) ? buffer.w & mask :
                    buffer.w >> 16);
            }

            float4 frag (v2f ps) : SV_Target
            {
                clip(ps.uv.z);
                uint2 px = round(ps.uv.xy * outRes);
                float4 col = flatten(px, uint2(px.y % 24,
                    floor(px.y / 24)) * 14).rrrr;
                return col;
            }
            ENDCG
        }
    }
}
