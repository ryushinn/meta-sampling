/*

This cooktorrance plugin implements the following BRDF equation:

fr(wi, wo) = kd / pi + ks * D(roughness, wi, wo) * G(wi, wo) * F(F0, wi, wo) / (pi * <N, wi> * <N, wo>).

And is from https://github.com/yongsen/mitsuba/blob/master/cooktorrance.cpp 

*/

#include <mitsuba/render/bsdf.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/core/warp.h>

MTS_NAMESPACE_BEGIN

class CookTorrance : public BSDF {
public:
	CookTorrance(const Properties &props)
		: BSDF(props) { 
	    m_diffuseReflectance = props.getSpectrum("diffuseReflectance", Spectrum(0.5f));
	    m_specularReflectance = props.getSpectrum("specularReflectance", Spectrum(0.2f));
	    m_roughness = props.getFloat("roughness", 0.1f);
	    m_F0 = props.getFloat("F0", 0.1f);
	}

	CookTorrance(Stream *stream, InstanceManager *manager)
		: BSDF(stream, manager) {
	    m_diffuseReflectance = Spectrum(stream);
	    m_specularReflectance = Spectrum(stream);
	    m_roughness = stream->readFloat();
	    m_F0 = stream->readFloat();

	    configure();
	}

	void configure() {
		m_components.clear();
		m_components.push_back(EGlossyReflection | EFrontSide );
		m_components.push_back(EDiffuseReflection | EFrontSide );
		m_usesRayDifferentials = false;

		Float dAvg = m_diffuseReflectance.getLuminance(),
		      sAvg = m_specularReflectance.getLuminance();
		m_specularSamplingWeight = sAvg / (dAvg + sAvg);

		BSDF::configure();
	}

	Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
	        /* sanity check */
	        if(measure != ESolidAngle || 
		   Frame::cosTheta(bRec.wi) <= 0 ||
	           Frame::cosTheta(bRec.wo) <= 0)
		  return Spectrum(0.0f);

  	        /* which components to eval */
		bool hasSpecular = (bRec.typeMask & EGlossyReflection)
				&& (bRec.component == -1 || bRec.component == 0);
		bool hasDiffuse  = (bRec.typeMask & EDiffuseReflection)
				&& (bRec.component == -1 || bRec.component == 1);

		/* eval spec */
		Spectrum result(0.0f);
		if (hasSpecular) {
		        Vector H = normalize(bRec.wo+bRec.wi);
			if(Frame::cosTheta(H) > 0.0f)
			{
			  // evaluate NDF
			  const Float roughness2 = m_roughness*m_roughness;
			  const Float cosTheta2 = Frame::cosTheta2(H);
			  const Float Hwi = dot(bRec.wi, H);
			  const Float Hwo = dot(bRec.wo, H);

			  const Float D = math::fastexp(-Frame::tanTheta2(H)/roughness2) / (roughness2 * cosTheta2*cosTheta2); 


			  // compute shadowing and masking
			  const Float G = std::min(1.0f, std::min( 
						   2.0f * Frame::cosTheta(H) * Frame::cosTheta(bRec.wi) / Hwi, 
						   2.0f * Frame::cosTheta(H) * Frame::cosTheta(bRec.wo) / Hwo          ));

			  // compute Fresnel
			  const Float F = fresnel(m_F0, Hwi);

			  // evaluate the microfacet model
			  result += m_specularReflectance * INV_PI * D * G * F / Frame::cosTheta(bRec.wi);
			}
		}

		/* eval diffuse */
		if (hasDiffuse)
		  result += m_diffuseReflectance * INV_PI * Frame::cosTheta(bRec.wo);

		// Done.
		return result;
	}

	Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
	        if (measure != ESolidAngle ||
			Frame::cosTheta(bRec.wi) <= 0 ||
			Frame::cosTheta(bRec.wo) <= 0 ||
			((bRec.component != -1 && bRec.component != 0) ||
			!(bRec.typeMask & EGlossyReflection)))
			return 0.0f;

		bool hasSpecular = (bRec.typeMask & EGlossyReflection)
				&& (bRec.component == -1 || bRec.component == 0);
		bool hasDiffuse  = (bRec.typeMask & EDiffuseReflection)
				&& (bRec.component == -1 || bRec.component == 1);

		Float diffuseProb = 0.0f, specProb = 0.0f;

		//* diffuse pdf */
		if (hasDiffuse)
			diffuseProb = warp::squareToCosineHemispherePdf(bRec.wo);

		/* specular pdf */
		if (hasSpecular) {
			Vector H = bRec.wo+bRec.wi;   Float Hlen = H.length();
			if(Hlen == 0.0f) specProb = 0.0f;
			else
			{
			  H /= Hlen;
			  const Float roughness2 = m_roughness*m_roughness;
			  const Float cosTheta2 = Frame::cosTheta2(H);
			  specProb = INV_PI * Frame::cosTheta(H) * math::fastexp(-Frame::tanTheta2(H)/roughness2) / (roughness2 * cosTheta2*cosTheta2) / (4.0f * absDot(bRec.wo, H));
			}
		}

		if (hasDiffuse && hasSpecular)
			return m_specularSamplingWeight * specProb + (1.0f-m_specularSamplingWeight) * diffuseProb;
		else if (hasDiffuse)
			return diffuseProb;
		else if (hasSpecular)
			return specProb;
		else
			return 0.0f;
	}

	Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &_sample) const {
	        Point2 sample(_sample);


		bool hasSpecular = (bRec.typeMask & EGlossyReflection)
				&& (bRec.component == -1 || bRec.component == 0);
		bool hasDiffuse  = (bRec.typeMask & EDiffuseReflection)
				&& (bRec.component == -1 || bRec.component == 1);


		if (!hasSpecular && !hasDiffuse)
			return Spectrum(0.0f);


		// determine which component to sample
		bool choseSpecular = hasSpecular;
		if (hasDiffuse && hasSpecular) {
			if (sample.x <= m_specularSamplingWeight) {
				sample.x /= m_specularSamplingWeight;
			} else {
				sample.x = (sample.x - m_specularSamplingWeight)
					/ (1.0f-m_specularSamplingWeight);
				choseSpecular = false;
			}
		}


		/* sample specular */
		if (choseSpecular) {
			Float cosThetaM = 0.0f, phiM = (2.0f * M_PI) * sample.y;
			Float tanThetaMSqr = -m_roughness*m_roughness * math::fastlog(1.0f - sample.x);
			cosThetaM = 1.0f / std::sqrt(1.0f + tanThetaMSqr);
			const Float sinThetaM = std::sqrt(std::max((Float) 0.0f, 1.0f - cosThetaM*cosThetaM));
			Float sinPhiM, cosPhiM;
			math::sincos(phiM, &sinPhiM, &cosPhiM);

			const Normal m = Vector(sinThetaM * cosPhiM,sinThetaM * sinPhiM,cosThetaM);

			// Perfect specular reflection based on the microsurface normal
			bRec.wo = 2.0f * dot(bRec.wi, m) * Vector(m) - bRec.wi;
			bRec.sampledComponent = 0;
			bRec.sampledType = EGlossyReflection;

	        /* sample diffuse */
		} else {
	   	        bRec.wo = warp::squareToCosineHemisphere(sample);
			bRec.sampledComponent = 1;
			bRec.sampledType = EDiffuseReflection;
		}
		bRec.eta = 1.0f;

		pdf = CookTorrance::pdf(bRec, ESolidAngle);

		/* unoptimized evaluation, explicit division of evaluation / pdf. */
		if (pdf == 0 || Frame::cosTheta(bRec.wo) <= 0)
			return Spectrum(0.0f);
		else
			return eval(bRec, ESolidAngle) / pdf;
	}

	Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
    	        Float pdf;
		return CookTorrance::sample(bRec, pdf, sample);
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		BSDF::serialize(stream, manager);

		m_diffuseReflectance.serialize(stream);
		m_specularReflectance.serialize(stream);
		stream->writeFloat( m_roughness );
		stream->writeFloat( m_F0 );
	}

	Float getRoughness(const Intersection &its, int component) const {
	       return m_roughness;
	}

	std::string toString() const {
	       std::ostringstream oss;
 	       oss << "Cook-Torrance[" << endl
	           << " id = \"" << getID() << "\"," << endl
		   << " diffuseReflectance = " << indent(m_diffuseReflectance.toString()) << ", " << endl
		   << " specularReflectance = " << indent(m_specularReflectance.toString()) << ", " << endl
		   << " F0 = " << m_F0 << ", " << endl
		   << " roughness = " << m_roughness << endl
		   << "]";
	       return oss.str();
	}

	Shader *createShader(Renderer *renderer) const;

	MTS_DECLARE_CLASS()
private:
	// helper method
	inline Float fresnel(const Float& F0, const Float& c) const
        {
	  return F0 + (1.0f - F0)*pow(1.0-c, 5.0f);
	}

	// attribtues
        Float m_F0;
        Float m_roughness;
        Spectrum m_diffuseReflectance;
        Spectrum m_specularReflectance;

        Float m_specularSamplingWeight;
};

// ================ Hardware shader implementation ================

/* CookTorrance shader-- render as a 'black box' */
class CookTorranceShader : public Shader {
public:
	CookTorranceShader(Renderer *renderer) :
		Shader(renderer, EBSDFShader) {
		m_flags = ETransparent;
	}

	void generateCode(std::ostringstream &oss,
			const std::string &evalName,
			const std::vector<std::string> &depNames) const {
		oss << "vec3 " << evalName << "(vec2 uv, vec3 wi, vec3 wo) {" << endl
			<< "    return vec3(0.0);" << endl
			<< "}" << endl;
		oss << "vec3 " << evalName << "_diffuse(vec2 uv, vec3 wi, vec3 wo) {" << endl
			<< "    return vec3(0.0);" << endl
			<< "}" << endl;
	}
	MTS_DECLARE_CLASS()
};

Shader *CookTorrance::createShader(Renderer *renderer) const {
	return new CookTorranceShader(renderer);
}

MTS_IMPLEMENT_CLASS(CookTorranceShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(CookTorrance, false, BSDF)
MTS_EXPORT_PLUGIN(CookTorrance, "CookTorrance BSDF");
MTS_NAMESPACE_END
