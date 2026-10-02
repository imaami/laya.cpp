// ROCm's single-precision exp and division, as its softmax kernels evaluate
// them. Pipelines using these must preserve the explicit FMAs
// (strict_spirv.hpp); precise keeps the remaining roundings in place.
float rocmExp(float x) {
    precise float scaled=x*uintBitsToFloat(0x3fb8aa3bu);
    float power=roundEven(scaled);
    precise float error=fma(x,uintBitsToFloat(0x3fb8aa3bu),-scaled);
    precise float fraction=scaled-power;
    error=fma(x,uintBitsToFloat(0x32a5705fu),error);
    fraction=fraction+error;
    float result=ldexp(exp2(fraction),int(power));
    if (x<uintBitsToFloat(0xc2ce8ed0u)) result=0.0;
    if (x>uintBitsToFloat(0x42b17218u)) result=uintBitsToFloat(0x7f800000u);
    return result;
}
float rocmDivide(float numerator,float denominator) {
    precise float reciprocal=1.0/denominator;
    reciprocal=fma(fma(-denominator,reciprocal,1.0),reciprocal,reciprocal);
    precise float quotient=numerator*reciprocal;
    precise float error=fma(-quotient,denominator,numerator);
    return fma(error,reciprocal,quotient);
}
