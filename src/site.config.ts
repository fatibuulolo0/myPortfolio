import { withBase } from "./utils/helpers";

export type Image = {
    src: string;
    alt?: string;
    caption?: string;
};

export type Link = {
    text: string;
    href: string;
};

export type Hero = {
    eyebrowText?: string;
    title?: string;
    text?: string;
    image?: Image;
    actions?: Link[];
};

export type About = {
    title?: string;
    text?: string;
};

export type Blog = {
    description?: string;
};

export type ContactInfo = {
    title?: string;
    text?: string;
    email?: {
        text?: string;
        href?: string;
        email?: string;
    };
    socialProfiles?: {
        text?: string;
        href?: string;
    }[];
};

export type Subscribe = {
    title?: string;
    text?: string;
    formUrl: string;
};

export type SiteConfig = {
    website: string;
    logo?: Image;
    title: string;
    description: string;
    image?: Image;
    headerNavLinks?: Link[];
    footerNavLinks?: Link[];
    socialLinks?: Link[];
    hero?: Hero;
    about?: About;
    contactInfo?: ContactInfo;
    subscribe?: Subscribe;
    blog?: Blog;
    postsPerPage?: number;
    recentPostLimit: number;
    projectsPerPage?: number;
};

const siteConfig: SiteConfig = {
    website: 'https://example.com',
    title: 'Portofolio',
    description: 'A minimal space-inspired personal blog template built with Astro.js and Tailwind CSS, by Siddhesh Thadeshwar',
    image: {
        src: '/space-ahead-preview.jpeg',
        alt: 'Space Ahead ✨ - A minimal space-inspired personal blog template, created by Siddhesh Thadeshwar.'
    },
    headerNavLinks: [
        {
            text: 'Home',
            href: withBase('/')
        },
        {
            text: 'About',
            href: withBase('/about')
        },
        {
            text: 'Skills',
            href: withBase('/skills')
        },
        {
            text: 'Projects',
            href: withBase('/projects')
        },
        {
            text: 'Contact',
            href: withBase('/contact')
        }
    ],   
    socialLinks: [
        {
            text: 'GitHub',
            href: 'https://github.com/FatiBuuloloo'
        },
        {
            text: 'Instagram',
            href: 'https://www.instagram.com/fatibuulolo/'
        },
        {
            text: 'LinkedIn',
            href: 'https://www.linkedin.com/in/fati-buulolo-7a9236391/'
        }
    ],
    hero: {
        //eyebrowText: 'Galaxy of Adventures',
        title: 'Fati Buulolo',
        text: "Mathematics Fresh Graduate",
        image: {
            src: '/assets/images/pixeltrue-space-discovery.svg',
            alt: 'A person sitting at a desk in front of a computer'
        },
        actions: [
            {
                text: 'Read Now',
                href: withBase('/blog')
            }          
        ]
    },
    about: {
        title: 'About',
        text: 'Mathematics Graduate | Data Science & Deep Learning Enthusiast. Recently graduated from Universitas Negeri Medan, I possess a deep interest in transforming complex data into actionable insights. My focus is on mastering Data Science and Machine Learning to achieve my goal of becoming a professional in the Data and AI field. I am eager to start my journey by contributing to impactful projects and solving real-world problems through advanced analytics.',
    },
    contactInfo: {
        title: 'Contact',
        text: "Hi! Whether you have a question, a suggestion, or just want to share your thoughts, I'm all ears. Feel free to get in touch through any of the methods below:",
        email: {
            text: "Drop me an email and I’ll do my best to respond as soon as possible.",
            href: "mailto:yahoo.com",
            email: "fatibuulolo2@yahoo.com"
        },
        socialProfiles: [
            {
                text: "LinkedIn",
                href: "https://www.linkedin.com/in/fati-buulolo-7a9236391/"
            },
            {
                text: "GitHub",
                href: "https://github.com/FatiBuuloloo"
            }
        ]
    },    
    blog: {
        description: "Read about my space adventures, explorations and the aliens I've met on my journeys."
    },
    postsPerPage: 2,
    recentPostLimit: 3
};

export default siteConfig;
